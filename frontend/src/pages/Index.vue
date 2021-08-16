<template>
  <q-page padding>
    <h5>{{ message }}</h5>
  </q-page>
</template>

<script>
import { defineComponent } from "vue";

export default defineComponent({
  name: "PageIndex",
  data() {
    return {
      message: 'I love Vue.js!'
    }
  }
});
</script>
