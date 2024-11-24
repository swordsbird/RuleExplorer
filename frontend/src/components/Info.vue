<template>
  <div class="info-container ma-3" :style="`position: absolute; ${positioning}; overflow-y: scroll`" v-resize="onResize">
    <div class="align-center tree-subtitle mb-2">
      Info
    </div>
    <div class="tree-text ml-1 mt-3" v-if="summary.info">
      <v-card elevation="0" class="mx-auto">
        <v-card-text class="text--primary pa-1 tree-text">
          <div>{{ view_info }}</div>
          <div>
            The current rules view shows 
            <b>{{summary.number_of_rules}}</b> rules covering 
            <b>{{summary.info.total}}</b> samples, including 
            <span :style="`color: ${color_scheme[1]}`"><b>{{summary.info.positives}} approved</b></span> samples and 
            <span :style="`color: ${color_scheme[0]}`"><b>{{summary.info.total - summary.info.positives}} rejected</b></span> samples, with a <span :style="`color: ${color_scheme[1]}`"><b>approval</b></span> rate of 
            <b>{{Number(summary.info.positives / summary.info.total * 100).toFixed(2)}}%</b>.
          </div>
          <div v-if="summary.show_update">
            <br/>
            After filtering, there are
            <b>{{summary.update.total}}</b> samples left with an <span :style="`color: ${color_scheme[1]}`"><b>approval</b></span> rate of 
            <b>{{Number(summary.update.positives / summary.update.total * 100).toFixed(2)}}%
              ({{summary.update.change_sgn}}{{Number(summary.update.changes * 100).toFixed(2)}}%)</b>.
              <!-- and an accuracy of 
              <b>{{Number(summary.update.accuracy * 100).toFixed(2)}}%</b>-->
          </div>
        </v-card-text>
      </v-card>
    </div>
    <template v-if="summary.current">
    <hr class="my-2"/>
      <div class="align-center tree-subtitle my-2">
        Rule
        <svg-icon
          v-if="summary.current.rule.hint_change != ''"
          type="mdi" class="mt-1"
          :path="glyph[summary.current.rule.hint_change].path">
        </svg-icon>
      </div>
        <v-card elevation="0" class="mx-auto">
          <!--v-card-title>Top 10 Australian beaches</v-card-title>
          <v-card-subtitle class="pb-0">
            Number 10
          </v-card-subtitle-->

          <v-card-text class="text--primary pa-1 tree-text">
            <div>
              <b>if</b>
              <v-chip label v-for="(item, i) in summary.current.items.filter(d => d.elements.length > 0)" :key="i"
                class="pa-1 ma-1"
              >
                {{item.text}}
              </v-chip>
              <b>then</b>
              <span :style="`color: ${color_scheme[summary.current.rule.predict]}`"><b>
                {{dataset.label[summary.current.rule.predict]}}
              </b></span>
            </div>
          </v-card-text>

          <!--v-card-actions>
            <v-btn
              text @click="updateRuleLabel({
                name: summary.current.id,
                label: 1
              })"
            >
              Anomaly
            </v-btn>

            <v-btn
              text @click="updateRuleLabel({
                name: summary.current.id,
                label: 0 
              })"
            >
              Non-anomaly
            </v-btn>
          </v-card-actions-->
        </v-card>
      <hr/>
    </template>
    <!--template v-if="zoom_level > 0">
      <div class="align-center tree-subtitle my-2">
        Suggestion
      </div>
      <template v-if="summary.suggestion">
        <div class="tree-text my-2" v-for="hint in summary.suggestion">
          If
          <b>{{hint[1][0]}} {{hint[1][1]}} {{Number(hint[1][2]).toFixed(6)}}</b>, the positive rate becomes
          <b>{{Number(hint[0] * 100).toFixed(2)}}%</b>
          <span style="color: green">&nbsp;(<b>+{{Number((hint[0] - summary.info.positives / summary.info.total) * 100).toFixed(2)}}%</b>)</span>
        </div>
      </template>
      <template v-if="!summary.suggestion">
        <div class="tree-text my-2" v-for="(target, index) in model_info.targets">
          <v-btn
            :loading="loading"
            class="ma-1"
            color="default"
            plain
            @click="findSuggestion(index)"
          >
            Increase
          </v-btn>
          <span
            style="font-style: italic; font-weight: 700;"
            :style="`color: ${color_scheme[index]}`">
              Prob({{target}})
          </span>
        </div>
      </template>
      <hr/>
    </template-->
    <div class="align-center tree-subtitle my-2 mt-4">
      Legend
    </div>
    <v-img
      class="mt-4"
      cover
      src="../assets/legend.png"
    ></v-img>
  </div>
</template>

<script>
// import * as vl from "vega-lite-api"
import { mapActions, mapState, mapGetters } from "vuex";
import SvgIcon from '@jamescoyle/vue-icon';

export default {
  name: "Feature",
  data() {
    return {
      info: null,
      hint: null,
      legend: null,
    };
  },
  components: {
    SvgIcon,
  },
  watch: {
    'summary.update'() {
      this.$forceUpdate()
    }
  },
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    },
    render: Boolean
  },
  computed: {
    ...mapState(["data_table", "data_header", "summary", 'model_info', 'dataset', 'glyph' ]),
    ...mapGetters([ 'rule_info', 'view_info', 'zoom_level' ]),
    color_scheme(){
      return this.dataset.color.color_scheme
    },
  },
  methods: {
    ...mapActions(['findSuggestion', 'updateRuleLabel']),
    async renderView() {
    },
    async onResize() {
      await this.renderView();
    }
  },
  mounted () {
    this.renderView()
  }
};
</script>
