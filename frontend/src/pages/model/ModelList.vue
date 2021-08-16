<template>
  <q-page padding>
    <div class="lm">
      <q-card bordered class="my-card">
        <q-card-section class="bg-primary">
          <div class="row items-center no-wrap">
            <div class="col">
              <div class="text-h6">BERT</div>
              <div class="text-subtitle2">Dec 12, 2019</div>
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
          <q-btn flat no-caps>Finetune</q-btn>
          <q-btn flat no-caps>Deploy</q-btn>
        </q-card-actions>
      </q-card>
    </div>

    <div>
      <q-card bordered class="my-card">
        <q-card-section class="bg-secondary">
          <div class="row items-center no-wrap">
            <div class="col">
              <div class="text-h6">GPT3</div>
              <div class="text-subtitle2">May 28, 2020</div>
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
          <q-btn flat no-caps>Finetune</q-btn>
          <q-btn flat no-caps>Deploy</q-btn>
        </q-card-actions>
      </q-card>
    </div>
  </q-page>
</template>

<script>
export default {
  setup() {
    return {
      bert: "Bidirectional Encoder Representations from Transformers (BERT) is a Transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google.[1][2] As of 2019, Google has been leveraging BERT to better understand user searches.[3]\n\nThe original English-language BERT has two models:[1] (1) the BERTBASE: 12 Encoders with 12 bidirectional self-attention heads, and (2) the BERTLARGE: 24 Encoders with 16 bidirectional self-attention heads. Both models are pre-trained from unlabeled data extracted from the BooksCorpus[4] with 800M words and English Wikipedia with 2,500M words",
      gpt3: "Generative Pre-trained Transformer 3 (GPT-3) is an autoregressive language model that uses deep learning to produce human-like text. It is the third-generation language prediction model in the GPT-n series (and the successor to GPT-2) created by OpenAI, a San Francisco-based artificial intelligence research laboratory.[2] GPT-3's full version has a capacity of 175 billion machine learning parameters. GPT-3, which was introduced in May 2020, and was in beta testing as of July 2020,[3] is part of a trend in natural language processing (NLP) systems of pre-trained language representations.[1] Before the release of GPT-3, the largest language model was Microsoft's Turing NLG, introduced in February 2020, with a capacity of 17 billion parameters - less than a tenth of GPT-3's.[4]",
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
