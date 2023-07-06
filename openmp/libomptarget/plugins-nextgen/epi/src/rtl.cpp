//===-RTLs/generic-64bit/src/rtl.cpp - Target RTLs Implementation - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL for generic 64-bit machine
//
//===----------------------------------------------------------------------===//

#include <string>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "Debug.h"
#include "omptargetplugin.h"

typedef struct TT{
	uint64_t remote;
	void *local;
	size_t size;
	struct TT *left, *right;
}TT;

typedef struct{
	int fd;
	int server;
	void *commandMem;
	int commandMemOrder;
	int commandMemId;
	TT *table;
}Device;

Device dev;
#ifndef TARGET_NAME
#define TARGET_NAME EPI Accelerator
#endif

#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

// RISC-V ELF ID
#define TARGET_ELF_ID 243

#include "../../common/elf_common/elf_common.h"

#define NUMBER_OF_DEVICES 1
#define OFFLOADSECTIONNAME ".omp_offloading.entries"

#ifdef __cplusplus
extern "C" {
#endif

static void *getCommandMem(size_t size){
	if(dev.commandMem == NULL){
		for(dev.commandMemOrder = 0; (1LLU << (dev.commandMemOrder + 12)) < size; dev.commandMemOrder++);
		dev.commandMem = mmap(NULL, (1LLU << (dev.commandMemOrder + 12)), PROT_READ | PROT_WRITE, MAP_SHARED, dev.fd, 0);
		read(dev.fd, &dev.commandMemId, 4);
	}else if((1LLU << (dev.commandMemOrder + 12)) < size){
		//todo
	}
	return dev.commandMem;
}

/*
 * Custom memcpy.
 * The default might cause alignement faults
 */
static void copyToTarget(void *dst, const void *src, size_t size){
	while((uint64_t)dst % 8){
		*((char *)dst) = *((const char *)src);
		dst = ((char *)dst) + 1;
		src = ((const char *)src) + 1;
		size--;
	}
	while(size >= 8){
		*((uint64_t *)dst) = *((const uint64_t *)src);
		dst = ((uint64_t *)dst) + 1;
		src = ((const uint64_t *)src) + 1;
		size -= 8;
	}
	
	while(size > 0){
		*((char *)dst) = *((const char *)src);
		dst = ((char *)dst) + 1;
		src = ((const char *)src) + 1;
		size--;
	}
	
	__sync_synchronize();
}

/*
 * Custom memcpy.
 * The default might cause alignement faults
 */
static void copyFromTarget(void *dst, const void *src, size_t size){
	while((uint64_t)src % 8){
		*((char *)dst) = *((const char *)src);
		dst = ((char *)dst) + 1;
		src = ((const char *)src) + 1;
		size--;
	}
	
	while(size >= 8){
		*((uint64_t *)dst) = *((const uint64_t *)src);
		dst = ((uint64_t *)dst) + 1;
		src = ((const uint64_t *)src) + 1;
		size -= 8;
	}
	
	while(size > 0){
		*((char *)dst) = *((const char *)src);
		dst = ((char *)dst) + 1;
		src = ((const char *)src) + 1;
		size--;
	}
	
	__sync_synchronize();
}


static void addTT(TT **table, uint64_t remote, void *local, size_t size){
	if(*table == NULL){
		*table = (TT *)malloc(sizeof(TT));
		(*table)->left = NULL;
		(*table)->right = NULL;
		(*table)->local = local;
		(*table)->remote = remote;
		(*table)->size = size;
	}else if(remote < (*table)->remote){
		addTT(&(*table)->left, remote, local, size);
	}else{
		addTT(&(*table)->right, remote, local, size);
	}
}

void *searchTT(TT *table, uint64_t remote){
	void *nextLocal;
	if(table == NULL){
		return NULL;
	}
	if(remote > table->remote){
		nextLocal = searchTT(table->right, remote);
		if(nextLocal == NULL){
			return table->local;
		}else{
			return nextLocal;
		}
	}else if(remote < table->remote){
		return searchTT(table->left, remote);
	}else{
		return table->local;
	}
}

static void addTranslation(int32_t device_id, uint64_t remote, void *local, size_t size){
	addTT(&dev.table, remote, local, size);
}


static void deleteTranslationHelper(TT **table){
	TT *current;
	
	current = *table;
	if(current->left == NULL){
		if(current->right == NULL){
			free(current);
			*table = NULL;
		}else{
			*table = current->right;
			free(current);
		}
	}else{
		if(current->right == NULL){
			*table = current->left;
			free(current);
		}else{
			TT *rmLeft;
			rmLeft = current->left;
			while(rmLeft->right != NULL){
				rmLeft = rmLeft->right;
			}
			rmLeft->right = current->right;
			*table = current->left;
			free(current);
		}
	}
}

static void *deleteTranslation(int32_t device_id, uint64_t remote, size_t *size){
	TT **current = &dev.table;
	while(*current != NULL){
		if((*current)->remote == remote){
			void *local;
			local = (*current)->local;
			*size = (*current)->size;
			deleteTranslationHelper(current);
			return local;
		}else if((*current)->remote > remote){
			current = &((*current)->right);
		}else{
			current = &((*current)->left);
		}
	}
	return NULL;
}

static void *translate(int32_t device_id, void *remote){
	return searchTT(dev.table, (uint64_t)remote);
}

int32_t __tgt_rtl_is_valid_binary(__tgt_device_image *Image) {
	return elf_check_machine(Image, TARGET_ELF_ID);
}

int32_t __tgt_rtl_number_of_devices(void) {
	return NUMBER_OF_DEVICES;
}

int32_t __tgt_rtl_init_device(int32_t ID) {
	dev.fd = open("/dev/omp-bridge", O_RDWR);
	if(dev.fd < 0){
		return OFFLOAD_FAIL;
	}
	dev.commandMem = NULL;
	dev.table = NULL;
	return OFFLOAD_SUCCESS;
}

__tgt_target_table *__tgt_rtl_load_binary(int32_t ID,
                                          __tgt_device_image *Image) {
  	size_t imgSize, entrySize, size;
	void *map, *imageMap, *entriesMap;
	__tgt_target_table *output;
	if(dev.fd < 0){
		return NULL;
	}
	imgSize = (size_t)Image->ImageEnd - (size_t)Image->ImageStart;
	entrySize = (size_t)Image->EntriesEnd - (size_t)Image->EntriesBegin;
	size = imgSize + entrySize + 2*(sizeof(size_t));
	map = getCommandMem(size);
	if(map == NULL){
		return NULL;
	}
	((size_t *)map)[0] = imgSize;
	((size_t *)map)[1] = entrySize;
	imageMap = (void *)((size_t)map + 2*(sizeof(size_t)));
	entriesMap = (void *)((size_t)imageMap + imgSize);
	
	memcpy(imageMap, Image->ImageStart, imgSize);
	memcpy(entriesMap, Image->EntriesBegin, entrySize);
	output = (__tgt_target_table *)malloc(sizeof(__tgt_target_table) + entrySize);
	output->EntriesBegin = (__tgt_offload_entry *)(output + 1);
	output->EntriesEnd = (__tgt_offload_entry *)((size_t)(output->EntriesBegin) + entrySize);
	__sync_synchronize();
	write(dev.fd, &dev.commandMemId, 8);
	memcpy(output->EntriesBegin, entriesMap, entrySize);
	output->EntriesBegin->name = NULL;
	return output;
}

void *__tgt_rtl_data_alloc(int32_t ID, int64_t Size, void *HostPtr, int32_t Kind) {
	uint64_t addr;
	void *map;
	int id;
	map = mmap(NULL, Size, PROT_READ | PROT_WRITE, MAP_SHARED, dev.fd, 0);
	if(map == MAP_FAILED){
		return NULL;
	}
	read(dev.fd, &id, 4);
	pread(dev.fd, &addr, 8, id);
	addTranslation(ID, addr, map, Size);
	return (void *)addr;
}

int32_t __tgt_rtl_data_submit(int32_t ID, void *TargetPtr, void *HostPtr,
                              int64_t Size) {
	void *local_tgt;
	local_tgt = translate(ID, TargetPtr);
	if(local_tgt == NULL){
		return OFFLOAD_FAIL;
	}
	copyToTarget(local_tgt, HostPtr, Size);
	return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_retrieve(int32_t ID, void *HostPtr, void *TargetPtr,
                                int64_t size) {
	void *local_tgt;
	local_tgt = translate(ID, TargetPtr);
	if(local_tgt == NULL){
		return OFFLOAD_FAIL;
	}
	copyFromTarget(HostPtr, local_tgt, size);
	return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_data_delete(int32_t ID, void *TargetPtr) {
	void *local;
	size_t size;
	local = deleteTranslation(ID, (uint64_t)TargetPtr, &size);
	if(local == NULL){
		return OFFLOAD_FAIL;
	}else{
		munmap(local, size);
		return OFFLOAD_SUCCESS;
	}
}

int32_t __tgt_rtl_run_target_team_region(int32_t ID, void *Entry, void **Args,
											ptrdiff_t *Offsets, int32_t NumArgs,
											int32_t NumTeams, int32_t ThreadLimit,
											uint64_t loop_tripcount){
	
	void *map;
	uint64_t msg;
	
	
	map = getCommandMem(sizeof(int64_t) + sizeof(void *) + NumArgs*(sizeof(void *) + sizeof(ptrdiff_t)));
	*((volatile int64_t *)(map)) = NumArgs;
	*((volatile void **)(((uint64_t)(map)) + sizeof(int64_t))) = Entry;
	memcpy((void *)((uint64_t)(map) + sizeof(int64_t) + sizeof(void *)), Offsets, sizeof(ptrdiff_t)*NumArgs);
	memcpy((void *)((uint64_t)(map) + sizeof(int64_t) + sizeof(void *) + sizeof(ptrdiff_t)*NumArgs), Args, sizeof(void *)*NumArgs);
	msg = (1LLU << 44) | dev.commandMemId;
	__sync_synchronize();
	write(dev.fd, &msg, 8);
	return OFFLOAD_SUCCESS;
}

int32_t __tgt_rtl_run_target_region(int32_t ID, void *Entry, void **Args,
									ptrdiff_t *Offsets, int32_t NumArgs) {
	
	void *map;
	uint64_t msg;
	
	
	map = getCommandMem(sizeof(int64_t) + sizeof(void *) + NumArgs*(sizeof(void *) + sizeof(ptrdiff_t)));
	*((volatile int64_t *)(map)) = NumArgs;
	*((volatile void **)(((uint64_t)(map)) + sizeof(int64_t))) = Entry;
	memcpy((void *)((uint64_t)(map) + sizeof(int64_t) + sizeof(void *)), Offsets, sizeof(ptrdiff_t)*NumArgs);
	memcpy((void *)((uint64_t)(map) + sizeof(int64_t) + sizeof(void *) + sizeof(ptrdiff_t)*NumArgs), Args, sizeof(void *)*NumArgs);
	msg = (1LLU << 44) | dev.commandMemId;
	__sync_synchronize();
	write(dev.fd, &msg, 8);
	return OFFLOAD_SUCCESS;
}


int32_t __tgt_rtl_is_data_exchangable(int32_t SrcDevId, int32_t DstDevId) {
	return 0;
}

int64_t __tgt_rtl_init_requires(int64_t RequiresFlags) {
	return OFFLOAD_SUCCESS;
}



int32_t __tgt_rtl_data_submit_async(int32_t ID, void *TargetPtr, void *HostPtr,
									int64_t Size,
									__tgt_async_info *AsyncInfoPtr) {
	return __tgt_rtl_data_submit(ID, TargetPtr, HostPtr, Size);
}

int32_t __tgt_rtl_data_retrieve_async(int32_t ID, void *HostPtr,
										void *TargetPtr, int64_t Size,
										__tgt_async_info *AsyncInfoPtr) {
	return __tgt_rtl_data_retrieve(ID, HostPtr, TargetPtr, Size);
}

int32_t __tgt_rtl_data_exchange(int32_t SrcID, void *SrcPtr, int32_t DstID,
								void *DstPtr, int64_t Size){
	return OFFLOAD_FAIL;
}

int32_t __tgt_rtl_data_exchange_async(int32_t SrcID, void *SrcPtr,
										int32_t DesID, void *DstPtr, int64_t Size,
										__tgt_async_info *AsyncInfoPtr){
	return OFFLOAD_FAIL;
}

int32_t __tgt_rtl_run_target_region_async(int32_t ID, void *Entry, void **Args,
											ptrdiff_t *Offsets, int32_t NumArgs,
											__tgt_async_info *AsyncInfoPtr) {
	return __tgt_rtl_run_target_region(ID, Entry, Args, Offsets, NumArgs);
}

// Asynchronous version of __tgt_rtl_run_target_team_region
int32_t __tgt_rtl_run_target_team_region_async(
	int32_t ID, void *Entry, void **Args, ptrdiff_t *Offsets, int32_t NumArgs,
	int32_t NumTeams, int32_t ThreadLimit, uint64_t loop_tripcount,
	__tgt_async_info *AsyncInfoPtr) {
	return __tgt_rtl_run_target_team_region(ID, Entry, Args, Offsets, NumArgs, NumTeams, ThreadLimit, loop_tripcount);
}


int32_t __tgt_rtl_synchronize(int32_t ID, __tgt_async_info *AsyncInfoPtr) {
	return OFFLOAD_SUCCESS;
}
#ifdef __cplusplus
}
#endif
