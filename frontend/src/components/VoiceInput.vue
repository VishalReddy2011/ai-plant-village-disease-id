<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import { useLanguageStore } from '@/stores/language'

const languageStore = useLanguageStore()

const LANG_MAP: Record<string, string> = {
  EN: 'en-US',
  HI: 'hi-IN',
  BN: 'bn-IN',
  TA: 'ta-IN',
  TE: 'te-IN',
  KN: 'kn-IN',
  MR: 'mr-IN',
  GU: 'gu-IN',
  PA: 'pa-IN',
  ML: 'ml-IN',
  OR: 'or-IN',
}

const isRecording = ref(false)
const transcript = ref('')
const voiceSubText = ref('Tap to describe your plant issue in your language')

const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
const isSupported = ref(!!SpeechRecognition)

let recognition: any = null

if (isSupported.value && SpeechRecognition) {
  recognition = new SpeechRecognition()
  recognition.continuous = false
  recognition.interimResults = false

  recognition.onstart = () => {
    isRecording.value = true
    transcript.value = ''
    voiceSubText.value = 'Listening... Speak now.'
  }

  recognition.onresult = (event: any) => {
    const result = event.results[0][0].transcript
    transcript.value = result
    voiceSubText.value = `"${result}"`
  }

  recognition.onerror = (event: any) => {
    console.error('Speech recognition error:', event.error)
    isRecording.value = false
    if (event.error === 'no-speech') {
      voiceSubText.value = 'No speech detected. Tap to try again.'
    } else if (event.error === 'not-allowed') {
      voiceSubText.value = 'Microphone permission denied. Please allow microphone access.'
    } else {
      voiceSubText.value = 'Could not hear you. Tap to try again.'
    }
  }

  recognition.onend = () => {
    isRecording.value = false
  }
}

const toggleVoice = () => {
  if (!isSupported.value) {
    voiceSubText.value = 'Speech recognition is not supported in this browser. Please try Chrome, Safari, or Edge.'
    return
  }

  if (isRecording.value) {
    recognition.stop()
  } else {
    const langCode = languageStore.currentLanguage[1]
    recognition.lang = LANG_MAP[langCode] || 'en-US'
    try {
      recognition.start()
    } catch (err) {
      console.error(err)
    }
  }
}

// Reset transcript explanation when language changes
watch(() => languageStore.currentLanguage, () => {
  transcript.value = ''
  voiceSubText.value = 'Tap to describe your plant issue in your language'
})

// Bridge global onclick call with Vue methods for compatibility with external triggers
onMounted(() => {
  ;(window as any).toggleVoice = toggleVoice
})

onBeforeUnmount(() => {
  if (recognition && isRecording.value) {
    recognition.stop()
  }
  if ((window as any).toggleVoice === toggleVoice) {
    delete (window as any).toggleVoice
  }
})
</script>

<template>
  <div class="w-full max-w-xl mx-auto px-4 mb-8">
    <div 
      class="voice-row flex items-center justify-between gap-5 p-5 bg-[#070c19]/70 border border-green-950/45 rounded-3xl shadow-sm transition-all duration-300 hover:border-green-900/35"
      :class="{ 'border-green-500/70 bg-green-950/10 shadow-lg shadow-green-500/5': isRecording }"
    >
      <div class="flex items-center gap-4.5">
        <!-- Microphone Button -->
        <button 
          @click="toggleVoice"
          id="voiceBtn"
          class="voice-btn flex items-center justify-center w-12 h-12 rounded-2xl cursor-pointer transition-all duration-300 focus:outline-none border select-none"
          :class="{ 'active': isRecording }"
          :style="isRecording 
            ? 'background: rgb(225, 245, 238); border-color: rgb(225, 245, 238); box-shadow: 0 10px 15px -3px rgba(34, 197, 94, 0.25); transform: scale(1.05);' 
            : 'background: rgba(29, 158, 117, 0.08); border-color: rgba(29, 158, 117, 0.2);'"
          aria-label="Toggle voice input"
        >
          <!-- Using standard Tabler icons tag loaded from our head stylesheet -->
          <i 
            class="ti ti-microphone" 
            :style="{ 
              fontSize: '18px', 
              color: isRecording ? 'rgb(29, 158, 117)' : 'rgb(45, 212, 151)',
              transition: 'all 0.3s ease'
            }" 
            id="voiceIcon" 
            aria-hidden="true"
          ></i>
        </button>

        <!-- Voice Text Information -->
        <div class="voice-text space-y-1">
          <div class="voice-title text-sm font-bold text-white tracking-wide">Voice Input</div>
          <div 
            class="voice-sub text-xs transition-colors duration-250 font-medium" 
            id="voiceSub"
            :class="[
              isRecording 
                ? 'text-green-400 font-semibold' 
                : transcript 
                  ? 'text-green-400 italic' 
                  : 'text-slate-400'
            ]"
          >
            {{ voiceSubText }}
          </div>
        </div>
      </div>

      <!-- Animated Audio Wave Visualizer -->
      <div 
        class="voice-waves pr-2" 
        id="voiceWaves" 
        :style="{ display: isRecording ? 'flex' : 'none' }"
      >
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
        <div class="wave"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.voice-waves {
  display: flex;
  align-items: center;
  gap: 3.5px;
  height: 24px;
}
.wave {
  width: 3.5px;
  height: 6px;
  background-color: rgb(29, 158, 117);
  border-radius: 9999px;
  animation: bounce 0.8s ease-in-out infinite alternate;
}
.wave:nth-child(2) {
  animation-delay: 0.15s;
}
.wave:nth-child(3) {
  animation-delay: 0.3s;
}
.wave:nth-child(4) {
  animation-delay: 0.45s;
}
.wave:nth-child(5) {
  animation-delay: 0.6s;
}

@keyframes bounce {
  from {
    height: 6px;
    transform: scaleY(0.7);
  }
  to {
    height: 24px;
    transform: scaleY(1.3);
  }
}
</style>