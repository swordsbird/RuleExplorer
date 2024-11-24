<template>
  <div
    ref="container" class="white"
    :style="`position: absolute; ${positioning}; user-select: none`"
    v-resize="onResize"
  >
    <!--v-btn
      style="position: absolute; top: 0px; right: 160px"
      :color="showShaps ? 'primary': 'grey'" text small
      @click="toggleShowShaps"
      >
      shap
    </v-btn-->
    <svg ref="tableview" style="width: 100%; height: 100%"></svg> 
  </div>
</template>

<script>
import * as d3 from 'd3'
import { mapActions, mapGetters, mapState } from 'vuex'
import SVGTable from '../libs/svgtable'
import utils from "../libs/utils"
export default {
  name: 'DataTable',
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    },
    display: Boolean
  },
  data: () => ({
    showShaps: false,
    instance: null,
    is_rendering: false,
  }),
  computed: {
    ...mapGetters(['filtered_data', 'zoom_level']),
    ...mapState(['debug', 'dataset', 'summary', 'highlighted_sample', 'covered_samples', 'crossfilter', 'data_table'])
  },
  watch: {
    covered_samples() { this.renderTable() },
    crossfilter () { this.renderTable() },
    highlighted_sample () { this.renderTable() },
    'summary.current'(){ this.renderTable() }
  },
  methods: {
    ...mapActions(['highlightSample', 'updateDataInfo']),
    async renderTable () {
      if (this.is_rendering) {
        return
      }
      this.is_rendering = 1
      const width = this.$refs.container.getBoundingClientRect().width - 4
      const height = this.$refs.container.getBoundingClientRect().height - 4

      const svg = d3.select(this.$refs.tableview)
      svg.selectAll('*').remove()

      if (!this.display) return
      let all_data = this.filtered_data // this.zoom_level ? this.filtered_data : this.data_table
      let reordered_data = null
      if (this.summary && this.summary.current) {
        let samples = [...this.summary.current.samples]
        reordered_data = all_data.filter(d => samples.indexOf(d._id) != -1)
      } else {
        reordered_data = all_data
        await this.updateDataInfo(reordered_data)
      }
      // console.log('reordered_data', reordered_data)

      if (this.highlighted_sample) {
        reordered_data = reordered_data.filter(d => this.highlighted_sample === d._id)
          .concat(reordered_data.filter(d => this.highlighted_sample !== d._id))
      }

      function getWeightedColor (baseColor, shap) {
        return d3.interpolateLab('white', baseColor)(Math.abs(shap) * 0.5)
      }

      if (reordered_data.length == 0) {
        this.is_rendering = 0
        return
      }

      const columns =
        Object.keys(reordered_data[0])
          .filter(d => d != '_id')
          .map(d => ({
            name: d,
            display: utils.naturallyShorten(d, 20),
            format: `,.${this.dataset.format}f`,
            isNumber: typeof reordered_data[0][d] === "number",
            width: 100,
          }))
      const self = this
      this.instance = new SVGTable(svg)
        .size([width, height])
        .fixedRows(this.highlighted_sample !== undefined ? 1 : 0)
        .fixedColumns(1)
        .rowsPerPage(50)
        .columns(columns)
        .style({ border: true })
        .cellRender(function (rect, fill, isHeader, isFixedRow, isFixedCol) {
          if (isHeader) return false
          if (isFixedCol) return false
          rect.attr('fill', d => {
            if (d.colIndex < 3) return 'white'
            return 'white'
          })
          return true
        })
        .highlightRender(function (r) {
          r.attr('fill', item => {
            const { d, hl } = item
            return hl ? d3.interpolateLab('grey', 'white')(0.8) : 'white'
          })
        })
        .data(reordered_data)
        .onclick(function (ctx, cell) {
          let sample_id = this._data[cell.rowIndex]._id
          self.highlightSample(sample_id)
        })
        .render()
      this.is_rendering = 0
    },
    toggleShowShaps () {
      if (this.instance) {
        this.instance.refresh()
      }
    },
    onResize () {
      this.renderTable()
    }
  }
}
</script>

<style>

</style>