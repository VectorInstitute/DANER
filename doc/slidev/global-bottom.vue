<template>
  <!-- <footer
    v-if="$slidev.nav.currentLayout !== 'cover'"
    class="absolute bottom-0 left-0 right-0 p-2"
  >
    Author: Yongchao Zhou
  </footer> -->

  <footer
    v-if="$slidev.nav.currentLayout !== 'cover' && $slidev.nav.currentPage !== $slidev.nav.total+1 "
    class="absolute bottom-0 left-0 right-0" 
  >
    <div class="relative bottom-0 p-4 float-right">
      {{ $slidev.nav.currentPage }}
    </div>
    <img src="vector_content.png" />
  </footer>

  <div v-if="$slidev.nav.currentLayout === 'end'">
    <img src="vector_thanks.png" />
  </div>
  
</template>

