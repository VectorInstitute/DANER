<template>
  <q-card>
    <q-card-section class="bg-primary text-white q-pa-sm" align="center">
      <div class="text-h4">Annotation</div>
    </q-card-section>
    <q-card-section class="flex justify-between">
      <div>
        <q-btn
          dense
          round
          color="primary"
          icon="arrow_back"
          class="q-ma-sm"
          @click="retrievePrev"
        />
        <q-btn
          dense
          round
          color="primary"
          icon="arrow_forward"
          class="q-ma-sm"
          @click="retrieveNext"
        />
      </div>
      <div>
        <q-btn
          dense
          round
          color="primary"
          icon="done"
          class="q-ma-sm"
          @click="onSubmitAnnotation"
        />
        <q-btn
          dense
          round
          color="primary"
          icon="close"
          class="q-ma-sm"
          @click="onRejectAnnotation"
        />
      </div>
    </q-card-section>

    <q-card-section>
      <div class="q-gutter-sm">
        <q-radio
          v-model="curEnt"
          keep-color
          v-for="(ent, index) in entList"
          :key="index"
          :val="ent.val"
          :label="ent.label"
          :color="ent.color"
        />
      </div>
    </q-card-section>

    <q-separator />

    <q-card-section>
      <token-render
        v-for="(content, index) in contentList"
        :key="index"
        :id="content.id"
        :token="content"
        @Mousedown="handleMouseDown"
        @Mouseup="handleMouseUp"
      ></token-render>
    </q-card-section>
  </q-card>
</template>

<script>
import { useQuasar } from "quasar";
import { computed } from "vue";
import { useStore } from "vuex";
import tokenRender from "src/components/annotation/ner/TokenRender.vue";

export default {
  components: { tokenRender },

  setup() {
    const $q = useQuasar();
    const store = useStore();

    const startIndex = computed({
      get: () => store.state.annotationNer.startIndex,
      set: (val) => {
        store.commit("annotationNer/updateStartIndex", val);
      },
    });

    const curEnt = computed({
      get: () => store.state.annotationNer.curEnt,
      set: (val) => {
        store.commit("annotationNer/updateCurEnt", val);
      },
    });

    const result = computed(() => store.state.annotationNer.result);
    const entList = computed(() => store.getters["annotationNer/getEntList"]);
    const contentList = computed(
      () => store.getters["annotationNer/getContentList"]
    );

    function handleMouseDown(evt) {
      const target = evt.currentTarget;
      startIndex.value = target.id;
    }

    function handleMouseUp(evt) {
      const target = evt.currentTarget;
      if (startIndex.value === target.id) {
        store.commit("annotationNer/toggleLabel", target.id);
      } else {
        store.commit("annotationNer/addLabel", {
          endIndex: target.id,
        });
      }
      startIndex.value = null;
      if (window.getSelection) {
        window.getSelection().removeAllRanges();
      } else if (document.selection) {
        document.selection.empty();
      }
    }

    function onSubmitAnnotation() {
      $q.notify({
        position: "top",
        spinner: true,
        message: "Send to backend",
        timeout: 1000,
      });

      store.dispatch("annotationNer/updateDataset");
    }

    function onRejectAnnotation() {
      store.dispatch("annotationNer/getDataset");
    }

    function retrievePrev() {
      store.commit("annotationNer/retrieveHistoryPrev");
    }

    function retrieveNext() {
      store.commit("annotationNer/retrieveHistoryNext");
    }

    // Get history

    return {
      curEnt,
      result,
      entList,
      contentList,
      handleMouseDown,
      handleMouseUp,
      onSubmitAnnotation,
      onRejectAnnotation,
      retrievePrev,
      retrieveNext,
    };
  },
};
</script>

<style lang="scss" scoped>
.label-entity {
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  font-size: 0.7em;
  line-height: 1;
  padding: 0.35em 0.35em;
  border-radius: 0.35em;
  text-transform: uppercase;
  display: inline-block;
  vertical-align: middle;
  margin: 0 0 0.15rem 0.5rem;
  background: #fff;
  font-weight: bold;
}
</style>
