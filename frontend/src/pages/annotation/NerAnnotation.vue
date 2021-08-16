<template>
  <q-page padding>
    <div class="myContainer">
      <ner-configuration></ner-configuration>
      <ner-annotation-render v-if="startAnnotation"> </ner-annotation-render>
    </div>
  </q-page>
</template>

<script>
import { useQuasar } from "quasar";
import { computed } from "vue";
import { useStore } from "vuex";
import NerAnnotationRender from "src/components/annotation/ner/NerAnnotationRender.vue";
import NerConfiguration from "src/components/annotation/ner/NerConfiguration.vue";

export default {
  components: { NerAnnotationRender, NerConfiguration },
  setup() {
    const $q = useQuasar();
    const store = useStore();

    const startAnnotation = computed(() => store.state.annotationNer.startAnnotation);

    return {
      startAnnotation
    };
  },
};
</script>

<style lang="scss" scoped>
.myContainer {
  max-width: 784px;
  margin: auto;
}

.q-card {
  margin: 3em 0em;
}
</style>
