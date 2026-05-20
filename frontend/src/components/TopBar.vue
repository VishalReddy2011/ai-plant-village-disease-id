<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { RouterLink } from 'vue-router'
import LeafIcon from './icons/LeafIcon.vue'
import { useLanguageStore } from '@/stores/language'

const languageStore = useLanguageStore()
const cur_lang = computed(() => languageStore.currentLanguage)
const isOpen = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)

const INDIAN_LANGUAGES: readonly [string, string][] = [
  ['हिन्दी', 'HI'],
  ['বাংলা', 'BN'],
  ['தமிழ்', 'TA'],
  ['తెలుగు', 'TE'],
  ['ಕನ್ನಡ', 'KN'],
  ['मराठी', 'MR'],
  ['ગુજરાતી', 'GU'],
  ['ਪੰਜਾਬੀ', 'PA'],
  ['മലയാളം', 'ML'],
  ['ଓଡ଼िଆ', 'OR'],
]

const selectLanguage = (lang: [string, string]) => {
  languageStore.setLanguage(lang)
  isOpen.value = false
}

const handleClickOutside = (event: MouseEvent) => {
  if (dropdownRef.value && !dropdownRef.value.contains(event.target as Node)) {
    isOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onBeforeUnmount(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<template>
  <nav class="flex justify-between items-center p-4 border-b border-green-950/35 bg-[#070c19]/85 backdrop-blur-md sticky top-0 z-40">
    <RouterLink to="/" class="no-underline">
      <div class="topbar-right flex items-center gap-3">
        <div class="logo-icon bg-green-950/30 p-2.5 rounded-xl text-green-400 shadow-sm border border-green-900/30">
          <LeafIcon />
        </div>
        <div class="logo-text">
          <div class="logo-heading text-lg font-bold text-white leading-tight">PlantVillage AI</div>
          <div class="logo-sub text-xs text-slate-400 font-medium">Disease Detection System</div>
        </div>
      </div>
    </RouterLink>
    <!-- <RouterLink to="/about"> About </RouterLink> -->
    <div class="relative" ref="dropdownRef">
      <button 
        class="btn-lang flex items-center gap-2.5 px-4 py-2 bg-[#0e172a] hover:bg-green-950/30 text-green-400 font-bold rounded-xl border border-green-900/30 shadow-xs transition-all duration-200 cursor-pointer text-sm focus:outline-none focus:ring-2 focus:ring-green-950/35"
        @click="isOpen = !isOpen"
      >
        <span>{{ cur_lang[0] }}</span>
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          class="h-4 w-4 text-green-450 transition-transform duration-200" 
          :class="{ 'rotate-180': isOpen }" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      <!-- Dropdown Menu -->
      <transition
        enter-active-class="transition duration-100 ease-out"
        enter-from-class="transform scale-95 opacity-0"
        enter-to-class="transform scale-100 opacity-100"
        leave-active-class="transition duration-75 ease-in"
        leave-from-class="transform scale-100 opacity-100"
        leave-to-class="transform scale-95 opacity-0"
      >
        <div 
          v-if="isOpen" 
          class="absolute right-0 mt-2 w-52 bg-[#0e172a] border border-green-900/30 rounded-2xl shadow-xl py-1.5 z-50 max-h-72 overflow-y-auto focus:outline-none scrollbar-thin scrollbar-thumb-green-950"
        >
          <!-- English option -->
          <button
            @click="selectLanguage(['English', 'EN'])"
            class="w-full text-left px-4 py-2.5 hover:bg-green-950/20 text-slate-300 transition-colors flex items-center justify-between cursor-pointer text-sm font-medium"
            :class="{ 'bg-green-950/40 text-green-400 font-bold': cur_lang[1] === 'EN' }"
          >
            <span>English</span>
            <span v-if="cur_lang[1] === 'EN'" class="text-[10px] bg-green-900/40 text-green-400 px-2 py-0.5 rounded-md font-bold uppercase tracking-wider">EN</span>
            <span v-else class="text-[10px] text-slate-500 font-bold uppercase tracking-wider font-mono">EN</span>
          </button>

          <!-- Divider -->
          <div class="h-px bg-green-950/40 my-1"></div>

          <!-- Indian languages -->
          <button
            v-for="[langName, langCode] in INDIAN_LANGUAGES"
            :key="langCode"
            @click="selectLanguage([langName, langCode])"
            class="w-full text-left px-4 py-2.5 hover:bg-green-950/20 text-slate-300 transition-colors flex items-center justify-between cursor-pointer text-sm font-medium"
            :class="{ 'bg-green-950/40 text-green-400 font-bold': cur_lang[1] === langCode }"
          >
            <span>{{ langName }}</span>
            <span v-if="cur_lang[1] === langCode" class="text-[10px] bg-green-900/40 text-green-400 px-2 py-0.5 rounded-md font-bold uppercase tracking-wider">{{ langCode }}</span>
            <span v-else class="text-[10px] text-slate-500 font-bold uppercase tracking-wider font-mono">{{ langCode }}</span>
          </button>
        </div>
      </transition>
    </div>
  </nav>
</template>

<style scoped></style>
