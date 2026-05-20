import { ref } from 'vue'
import { defineStore } from 'pinia'

export const useLanguageStore = defineStore('language', () => {
  const currentLanguage = ref<[string, string]>(['English', 'EN'])

  const setLanguage = (lang: [string, string]) => {
    currentLanguage.value = lang
  }

  return {
    currentLanguage,
    setLanguage
  }
})
