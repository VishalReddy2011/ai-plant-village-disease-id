<script setup lang="ts">
import { RouterLink } from 'vue-router'
import LeafIcon from './icons/LeafIcon.vue'
let cur_lang: string[] = ['English', 'EN']
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
  ['ଓଡ଼ିଆ', 'OR'],
]
</script>

<template>
  <nav class="flex justify-between">
    <RouterLink to="/">
      <div class="topbar-right flex">
        <div class="logo-icon">
          <LeafIcon />
        </div>
        <div class="logo-text">
          <div class="logo-heading">PlantVillage AI</div>
          <div class="logo-sub">Disease Detection System</div>
        </div>
      </div>
    </RouterLink>
    <!-- <RouterLink to="/about"> About </RouterLink> -->
    <div class="topbar-left">
      <button class="btn-lang" v-on:click="">{{ cur_lang[0] }}</button>
    </div>
  </nav>
</template>

<style scoped></style>