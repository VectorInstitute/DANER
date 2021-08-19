<template>
  <q-page padding>
    <div class="myContainer">
      <q-card>
        <q-card-section class="bg-primary text-white" align="center">
          <div class="text-h4">Named Entity Recognition Service</div>
        </q-card-section>
        <div class="q-ma-md">
          <div class="q-gutter-sm">
            <q-checkbox
              v-model="ents"
              keep-color
              v-for="(ent, index) in ALL_ENTS"
              :key="index"
              :val="ent.val"
              :label="ent.label"
              :color="ent.color"
            />
          </div>
          <q-separator />
          <q-btn color="primary" label="Select All" @click="onSelectALL" />
          <q-btn
            color="secondary"
            label="Default"
            @click="onSelectDefault"
            class="q-ma-md"
          />
          <q-btn
            color="primary"
            label="Reset"
            flat
            @click="onResetLabel"
            class="q-ma-md"
          />
        </div>
        <div class="q-pa-md">
          <div class="q-gutter-md row items-start">
            <q-select
              filled
              v-model="model"
              :options="ALL_MODELS"
              label="Model"
              style="width: 250px"
            />
          </div>
        </div>
      </q-card>

      <q-card>
        <form
          @submit.prevent.stop="onSubmit"
          @reset.prevent.stop="onReset"
          class="q-gutter-md"
        >
          <div>
            <q-input
              ref="textRef"
              filled
              type="textarea"
              v-model="text"
              hint="Max characters 1500"
              counter
              lazy-rules
              autogrow
              :rules="textRules"
              :input-style="{ 'min-height': '100px' }"
            />
          </div>

          <div>
            <q-btn
              label="Analyze"
              type="submit"
              color="primary"
              class="q-ma-md"
            />
            <q-btn
              label="Reset"
              type="reset"
              color="primary"
              flat
              class="q-ma-md"
            />
          </div>
        </form>
      </q-card>

      <ner-render
        :nerText="text"
        :nerSpans="result"
        :ents="ents"
        :entStyle="ALL_ENTS"
      >
      </ner-render>
    </div>
  </q-page>
</template>

<script>
import { useQuasar } from "quasar";
import { ref, computed } from "vue";
import { useStore } from "vuex";
import NerRender from "src/components/service/ner/NerServiceRender.vue";

export default {
  components: { NerRender },
  setup() {
    const $q = useQuasar();
    const store = useStore();
    const ALL_MODELS = store.state.serviceNer.ALL_MODELS;
    const ALL_ENTS = store.state.serviceNer.ALL_ENTS;

    const textRef = ref(null);
    const text = computed({
      get: () => store.state.serviceNer.text,
      set: (val) => {
        store.commit("serviceNer/updateText", val);
      },
    });

    const ents = computed({
      get: () => store.state.serviceNer.ents,
      set: (val) => {
        store.commit("serviceNer/updateEnts", val);
      },
    });

    const model = computed({
      get: () => store.state.serviceNer.model,
      set: (val) => {
        store.commit("serviceNer/updateModel", val);
      },
    });

    const result = computed(() => store.state.serviceNer.result);

    const textRules = [
      (val) => (val && val.length > 0) || "Please type something to analyze!",
      (val) => (val && val.length < 1500) || "Text too long",
    ];

    function onSelectALL() {
      ents.value = store.getters["serviceNer/getAllEnts"];
    }
    function onSelectDefault() {
      ents.value = store.getters["serviceNer/getDefaultEnts"];
    }

    function onResetLabel() {
      ents.value = [];
    }

    function onSubmit() {
      textRef.value.validate();

      if (textRef.value.hasError) {
        $q.notify({
          color: "negative",
          message: "No content to analyze",
        });
      } else {
        $q.notify({
          icon: "done",
          color: "positive",
          message: "Send to server for analyze",
        });
      }
      store.dispatch("serviceNer/analyzeText", { textInput: text.value });
    }

    function onReset() {
      text.value = null;
      textRef.value.resetValidation();
    }

    return {
      text,
      textRef,
      textRules,
      result,
      ents,
      model,
      ALL_ENTS,
      ALL_MODELS,
      onSelectALL,
      onSelectDefault,
      onResetLabel,
      onSubmit,
      onReset,
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
