<script setup lang="ts">
import { ref, onBeforeUnmount } from 'vue'
import PhotoUpIcon from './icons/PhotoUpIcon.vue'
import UploadIcon from './icons/UploadIcon.vue'

// Allowed file settings based on the image
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
const MAX_FILE_SIZE_MB = 10
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

const isDragging = ref(false)
const selectedFile = ref<File | null>(null)
const previewUrl = ref<string | null>(null)
const errorMessage = ref<string | null>(null)

const fileInput = ref<HTMLInputElement | null>(null)

const triggerFileInput = () => {
  fileInput.value?.click()
}

const handleFileChange = (event: Event) => {
  const target = event.target as HTMLInputElement
  if (target.files && target.files.length > 0) {
    const file = target.files[0]
    if (file) {
      processFile(file)
    }
  }
}

const handleDragOver = (event: DragEvent) => {
  event.preventDefault()
  isDragging.value = true
}

const handleDragLeave = () => {
  isDragging.value = false
}

const handleDrop = (event: DragEvent) => {
  event.preventDefault()
  isDragging.value = false
  if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
    const file = event.dataTransfer.files[0]
    if (file) {
      processFile(file)
    }
  }
}

const processFile = (file: File) => {
  errorMessage.value = null
  
  if (!ALLOWED_TYPES.includes(file.type)) {
    errorMessage.value = 'Unsupported file format. Please upload JPEG, PNG, or WEBP.'
    return
  }
  
  if (file.size > MAX_FILE_SIZE_BYTES) {
    errorMessage.value = `File is too large. Max size is ${MAX_FILE_SIZE_MB}MB.`
    return
  }
  
  selectedFile.value = file
  // Create object URL for preview
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
  previewUrl.value = URL.createObjectURL(file)
}

const clearImage = () => {
  selectedFile.value = null
  errorMessage.value = null
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

const formatBytes = (bytes: number, decimals = 2) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

onBeforeUnmount(() => {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
})
</script>

<template>
  <div class="max-w-xl mx-auto my-8 px-4">
    <!-- Main Drop Zone -->
    <div
      @dragover="handleDragOver"
      @dragleave="handleDragLeave"
      @drop="handleDrop"
      @click="triggerFileInput"
      class="upload-zone relative flex flex-col items-center justify-center border-2 border-dashed rounded-3xl p-8 text-center cursor-pointer transition-all duration-300 overflow-hidden"
      :class="[
        isDragging
          ? 'border-green-500 bg-green-950/35 scale-[1.01] shadow-lg shadow-green-500/10'
          : errorMessage
            ? 'border-rose-900/40 bg-rose-950/20'
            : 'border-green-950/45 bg-[#070c19]/70 hover:bg-green-950/10 hover:border-green-800'
      ]"
    >
      <input
        type="file"
        ref="fileInput"
        class="hidden"
        accept="image/jpeg, image/png, image/webp"
        @change="handleFileChange"
      />

      <!-- When no image is uploaded -->
      <div v-if="!previewUrl" class="flex flex-col items-center gap-4">
        <!-- Icon Wrapper -->
        <div 
          class="upload-icon-wrap p-4 bg-green-950/30 rounded-2xl shadow-sm border border-green-900/30 text-green-400 transition-transform duration-300"
          :class="{ 'scale-110 rotate-3': isDragging }"
        >
          <PhotoUpIcon class="w-8 h-8" />
        </div>

        <!-- Info/Instructions -->
        <div class="space-y-1">
          <h3 class="upload-title text-base font-bold text-white">
            Drop your leaf image here
          </h3>
          <p class="upload-sub text-sm text-slate-400 font-medium">
            or click to browse from your device
          </p>
        </div>

        <!-- Supported types row -->
        <div class="badge-row flex flex-wrap justify-center gap-2 mt-2">
          <span class="badge flex items-center gap-1.5 px-3 py-1 bg-[#0d1527] border border-green-950/40 rounded-xl text-xs text-green-400 font-bold shadow-xs">
            <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span> JPEG
          </span>
          <span class="badge flex items-center gap-1.5 px-3 py-1 bg-[#0d1527] border border-green-950/40 rounded-xl text-xs text-green-400 font-bold shadow-xs">
            <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span> PNG
          </span>
          <span class="badge flex items-center gap-1.5 px-3 py-1 bg-[#0d1527] border border-green-950/40 rounded-xl text-xs text-green-400 font-bold shadow-xs">
            <span class="w-1.5 h-1.5 rounded-full bg-green-400"></span> WEBP
          </span>
          <span class="badge flex items-center gap-1.5 px-3 py-1 bg-green-950/40 border border-green-900/35 rounded-xl text-xs text-green-400 font-bold shadow-xs">
            Max 10 MB
          </span>
        </div>

        <!-- Action Button -->
        <button 
          type="button"
          class="btn-upload flex items-center gap-2 px-5 py-2.5 bg-green-600 hover:bg-green-500 text-white font-semibold rounded-2xl shadow-md shadow-green-900/20 transition-all duration-200 cursor-pointer text-sm border-0"
        >
          <UploadIcon class="w-4 h-4 text-white" />
          Choose Image
        </button>
      </div>

      <!-- Preview State -->
      <div v-else class="w-full flex flex-col items-center gap-5" @click.stop>
        <!-- Image Preview Frame -->
        <div class="relative group max-w-xs rounded-2xl overflow-hidden shadow-md border border-green-900/30 bg-[#0d1527] p-1">
          <img
            :src="previewUrl"
            alt="Leaf Preview"
            class="w-full h-48 object-cover rounded-xl transition-transform duration-300 group-hover:scale-[1.02]"
          />
          <!-- Remove overlay/button -->
          <button
            type="button"
            @click="clearImage"
            class="absolute top-3 right-3 p-2 bg-slate-900/80 hover:bg-slate-900 text-white rounded-full transition-all shadow-md backdrop-blur-xs cursor-pointer border-0"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <!-- File Details -->
        <div class="text-center space-y-1">
          <p class="text-sm font-bold text-slate-200 truncate max-w-xs">
            {{ selectedFile?.name }}
          </p>
          <p class="text-xs text-slate-400 font-medium">
            {{ formatBytes(selectedFile?.size || 0) }}
          </p>
        </div>

        <!-- Post-Upload Actions -->
        <div class="flex gap-3 justify-center w-full max-w-xs">
          <button
            type="button"
            @click="clearImage"
            class="flex-1 py-2.5 px-4 bg-[#0d1527] hover:bg-green-950/20 text-green-400 font-semibold rounded-2xl text-sm transition-all cursor-pointer border border-green-900/30"
          >
            Change
          </button>
          <button
            type="button"
            class="flex-1 py-2.5 px-4 bg-green-600 hover:bg-green-500 text-white font-semibold rounded-2xl text-sm transition-all shadow-md shadow-green-900/20 cursor-pointer border-0"
          >
            Analyze Leaf
          </button>
        </div>
      </div>
    </div>

    <!-- Error Message Panel -->
    <transition
      enter-active-class="transition duration-350 ease-out"
      enter-from-class="transform -translate-y-2 opacity-0"
      enter-to-class="transform translate-y-0 opacity-100"
      leave-active-class="transition duration-200 ease-in"
      leave-from-class="transform translate-y-0 opacity-100"
      leave-to-class="transform -translate-y-2 opacity-0"
    >
      <div
        v-if="errorMessage"
        class="mt-4 p-3.5 bg-rose-950/20 border border-rose-900/30 rounded-2xl flex items-start gap-2.5 animate-pulse"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-rose-450 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <span class="text-xs text-rose-400 font-semibold leading-normal">{{ errorMessage }}</span>
      </div>
    </transition>
  </div>
</template>

<style scoped></style>
