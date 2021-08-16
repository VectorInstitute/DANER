<template>
  <q-page padding>
    <div class="lm">
      <q-card bordered class="my-card">
        <q-card-section class="bg-primary">
          <div class="row items-center no-wrap">
            <div class="col">
              <div class="text-h6">CONLL 2003</div>
              <div class="text-subtitle2">News</div>
            </div>

            <div class="col-auto">
              <q-btn color="grey-7" round flat icon="more_vert">
                <q-menu cover auto-close>
                  <q-list>
                    <q-item clickable>
                      <q-item-section>Remove Card</q-item-section>
                    </q-item>
                    <q-item clickable>
                      <q-item-section>Send Feedback</q-item-section>
                    </q-item>
                    <q-item clickable>
                      <q-item-section>Share</q-item-section>
                    </q-item>
                  </q-list>
                </q-menu>
              </q-btn>
            </div>
          </div>
        </q-card-section>

        <q-card-section class="bg-grey-1">
          {{ bert }}
        </q-card-section>

        <q-separator />

        <q-card-actions>
          <q-btn flat no-caps>Investigation</q-btn>
          <q-btn flat no-caps>Annoation</q-btn>
          <q-btn flat no-caps>Quality Control</q-btn>
        </q-card-actions>
      </q-card>
    </div>

    <div>
      <q-card bordered class="my-card">
        <q-card-section class="bg-secondary">
          <div class="row items-center no-wrap">
            <div class="col">
              <div class="text-h6">GUM-3.1.0</div>
              <div class="text-subtitle2">Wiki</div>
            </div>

            <div class="col-auto">
              <q-btn color="grey-7" round flat icon="more_vert">
                <q-menu cover auto-close>
                  <q-list>
                    <q-item clickable>
                      <q-item-section>Remove Card</q-item-section>
                    </q-item>
                    <q-item clickable>
                      <q-item-section>Send Feedback</q-item-section>
                    </q-item>
                    <q-item clickable>
                      <q-item-section>Share</q-item-section>
                    </q-item>
                  </q-list>
                </q-menu>
              </q-btn>
            </div>
          </div>
        </q-card-section>

        <q-card-section class="bg-grey-1">
          {{ gpt3 }}
        </q-card-section>

        <q-separator />

        <q-card-actions>
          <q-btn flat no-caps>Investigation</q-btn>
          <q-btn flat no-caps>Annoation</q-btn>
          <q-btn flat no-caps>Quality Control</q-btn>
        </q-card-actions>
      </q-card>
    </div>
  </q-page>
</template>

<script>
export default {
  setup() {
    return {
      bert: "Dataset Details",
      gpt3: "Dataset Details",
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
  margin: auto;
  width: 100%;
  max-width: 768px;
}
.lm {
  margin: 3em 0em;
}
</style>
