<template>
  <div class="feature-container" :style="`position: absolute; ${positioning}; overflow-y: scroll`" v-resize="onResize">
    <div class="align-center tree-subtitle ma-2 ml-3 mt-3">
      Model
    </div>
    <div class="tree-text ma-3">
    <!--v-card-actions>
      <v-btn
        text @click="clearSession()"
      >
        Reset
      </v-btn>
    </v-card-actions-->
      
    <p class="ml-1"> <span style="font-weight: 400">Model: </span> {{ model_info.model }} </p>
    <p class="ml-1"> <span style="font-weight: 400">Dataset: </span> {{ model_info.dataset }} </p>
    <!--p class="ml-1"> <span style="font-weight: 400">Accuracy: </span> {{ Number(model_info.accuracy * 100).toFixed(2) }}% </p-->
    <p class="ml-1"> <span style="font-weight: 400">Fidelity: </span> {{ Number(model_info.info.fidelity_test * 100).toFixed(2) }}% </p>
    <p class="ml-1"> <span style="font-weight: 400">Num of rules: </span> {{ model_info.num_of_rules }} </p>
    </div>
    <hr/>
    <svg ref="feature_parent" style="width: 100%" class="my-2"></svg>
  </div>
</template>

<script>
// import * as vl from "vega-lite-api"
import { mapActions, mapGetters, mapState } from "vuex";
import * as d3 from "d3";
import BrushableBarchart from "../libs/brushablechart";

export default {
  name: "Feature",
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    },
    render: Boolean
  },
  computed: {
    ...mapState(["model_info", 'dataset', 'dataset_candidates', "covered_samples", "data_features", "data_table", "data_header", "featureview", "rules"]),
    ...mapGetters(['rule_related_data', 'zoom_level', 'model_target'])
  },
  data() {
    return {
      model_feature_view: null,
      data_feature_view: null,
      model_features: null,
      data_cols: null,
      current_dataset: 'bankruptcy',
    }
  },
  watch: {
    covered_samples(val) {
      this.update()
    }
  },
  methods: {
    ...mapActions(['tooltip', 'updateCrossfilter', 'updateRulefilter', 'clearSession']),
    renderView() {
      const width = this.$refs.feature_parent
        .parentNode
        .getBoundingClientRect()
        .width
      const self = this
      const featureview = this.featureview
      const data_cols = this.data_features
      /*Object.keys(this.data_table[0])
        .filter(d => d != 'id' && d != '_id')*/
        .map(d => ({ name: d.name, key: d.name, feature: d, filter: () => 1 }))
      const model_features = []
      /*
        { name: 'Confidence', key: 'fidelity', filter: () => 1 },
        { name: 'Coverage', key: 'coverage', filter: () => 1 },
        { name: 'Anomaly Score', key: 'LOF', filter: () => 1 },
      ]
      */
      let model_feature_height = 0//(model_features.length + 1) * featureview.column_height
      let data_feature_height = (data_cols.length + 1) * featureview.column_height
      
      const svg = d3.select(this.$refs.feature_parent)
      svg.selectAll("*").remove()
      const height = model_feature_height + data_feature_height
      svg.attr("height", height)
        .attr("width", width)

      let model_feature_view = svg.append('g')
        .attr('class', 'model-feature')
        .attr('transform', `translate(${0},${0})`)
        .attr('display', 'none')

      let data_feature_view = svg.append('g')
        .attr('class', 'data-feature')
        .attr('transform', `translate(${0},${model_feature_height})`)

      model_feature_view
        .append("text")
        .attr("class", 'tree-subtitle')
        .attr("dx", 10)
        .attr("dy", 30)
        .text('Model Attributes')
      
      data_feature_view
        .append("text")
        .attr("class", 'tree-subtitle')
        .attr("dx", 10)
        .attr("dy", 20)
        .text('Data Attributes')

      model_feature_view = model_feature_view.append('g')
        .attr('class', 'content')

      data_feature_view = data_feature_view.append('g')
        .attr('class', 'content')

      this.model_feature_view = model_feature_view
      this.data_feature_view = data_feature_view
      this.model_features = model_features
      this.data_cols = data_cols
      this.update()
    },
    update() {
      // console.log('update with', this.covered_samples, this.rule_related_data)
      const self = this
      const featureview = this.featureview
      const width = this.$refs.feature_parent
        .parentNode
        .getBoundingClientRect()
        .width

      let current_color = 'white'
      if (this.zoom_level) {
        current_color = this.dataset.color.color_scheme[this.rules[0].predict]
      }
      const model_target = this.model_target

      drawCharts(
        this.model_feature_view,
        this.rules,
        this.model_features,
        (filter) => this.updateRulefilter(filter), false
      )
      drawCharts(
        this.data_feature_view,
        this.zoom_level ? {
          first: this.data_table, second: this.rule_related_data
        } : this.data_table,//this.rule_related_data,
        this.data_cols, 
        (filter) => this.updateCrossfilter(filter)
      )

      function drawCharts(selection, data, features, update, brushable = true) {
        selection.selectAll('*').remove()

        selection
          .selectAll(".chart")
          .data(features)
          .enter()
          .append("g")
          .attr("class", "chart")
        
        let chart_row = selection
          .selectAll(".chart")
          .data(features)
          .attr("transform", (d, i) => `translate(${featureview.padding}, ${(i + 0.5) * featureview.column_height})`);

        chart_row
          .append("text")
          .attr("class", "name")
          .attr("dy", featureview.column_height - 10)
          .style("font-family", "Arial")
          .style("font-size", "16px")
          .style("font-weight", featureview.fontweight)
          .text((d) => (d.name.length < featureview.maxlen ? d.name : d.name.slice(0, featureview.maxlen) + "..."))
          .on("mouseover", function(ev, d){
            if (d.name.length < featureview.maxlen) return
            self.tooltip({ type: "show" })
          })
          .on("mouseout", function(ev, d){
            if (d.name.length < featureview.maxlen) return
            self.tooltip({ type: "hide" })
          })
          .on("mousemove", function(ev, d){
            if (d.name.length < featureview.maxlen) return
            self.tooltip({
              type: "text",
              data: `${d.name}`
            })
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
          })
          
        const chart_body = chart_row
          .append("g")
          .attr("class", "chart")
          .attr("transform", `translate(${featureview.textwidth}, 0)`);
        
        chart_body.each(function(d) {
          // console.log(d.name, d.feature.values)
          if (d.name == 'industry') {
            // console.log(d)
          }
          const chart = BrushableBarchart()
            .data(data)
            .x(d.key)
            .valueTicks(d.feature.values)
            .datatype(d.feature.dtype)
            .width(width - featureview.padding * 4 - featureview.textwidth - featureview.scrollbar_width)
            .height(featureview.chart_height)
            .target(model_target)
            .brushable(brushable)
            .colors({
              handle: featureview.handle_color,
              glyph: featureview.glyph_color,
              bar: featureview.bar_color,
              highlight: featureview.highlight_color,
              //d3.color(current_color).darker(-0.5)
            })
            .mousemove({
              first: function(ev, d){
                // console.log('info', d)
                const rate = Number(100 * Object.values(d.target)[0] / d.count).toFixed(2)
                self.tooltip({ type: "show" })
                self.tooltip({ type: "text", data: `${d.name}: <br/> ${d.count} instances, <br/> ${Object.keys(d.target).map(k => d.target[k] + ' ' + k).join(', ')}<br/>${rate}%`})
                self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
              },
              second: function(ev, d){
                if (data instanceof Array) return
                const rate = Number(100 * Object.values(d.target2)[0] / d.count2).toFixed(2)
                self.tooltip({ type: "show" })
                self.tooltip({ type: "text", data: `${d.name}: <br/> ${d.count2} instances, <br/> ${Object.keys(d.target2).map(k => d.target2[k] + ' ' + k).join(', ')}<br/>${rate}%`})
                self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY }})
              },
            })
            .mouseout(function(ev, d){
              self.tooltip({ type: "hide" })
            })

          chart.brushend(function(){
            d.filter = chart.filter()
            const filter = (d) => {
              for (let feature of features) {
                if (!feature.filter(d)) {
                  // console.log(feature.feature.name, feature.feature.values, d[feature.feature.name])
                  return 0
                }
              }
              return 1
            }
            update(filter)
          })

          d3.select(this).call(chart)
        });
      }
    },
    onResize() {
      this.renderView()
    },
  },
  mounted () {
    this.renderView()
  }
};
</script>
