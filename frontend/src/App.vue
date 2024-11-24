<template>
  <v-app>
    <v-system-bar app height="48">
      <div class="d-flex align-center text-h5 font-weight-medium">
        RuleExplorer
      </div>

      <v-spacer></v-spacer>
    </v-system-bar>

    <v-main>
      <v-container fluid style="height: 100%; position: relative" ref="main_container" class="pa-1">
        <v-row :style="`height: ${showTable ? 80 : 100}%`" dense>
          <v-col cols="2" class="pa-1">
            <v-card class="mx-auto" id="feature_view" height="100%">
              <div style="height: 100%">
                <feature v-if="initilized"/>
              </div>
            </v-card>
          </v-col>
          <v-col cols="8" class="pa-1">
            <v-card class="mx-auto" height="100%" style="user-select: none">
              <div
                :style="`height: 100%; position: relative;`"
                class="white"
              >
                <div style="height: calc(100%); position: relative;">
                  <matrix ref="matrix" v-if="initilized"/>
                </div>
                <!--
                div class="d-flex justify-space-between align-end pr-4" v-if="initilized">
                  <div>
                    <v-btn color="primary" text @click="toggleDatatable" v-show="false">
                      <span v-if="showTable">hide data</span>
                      <span v-else>show data</span>
                    </v-btn>
                  </div>
                  <div>
                    <div class="d-flex justify-end"></div>
                  </div>
                </div-->
              </div>
            </v-card>
          </v-col>
          <v-col cols="2" class="pa-1">
            <v-card class="mx-auto" height="100%">
              <div style="height: 100%; position: relative;">
                <info :render="initilized"/>
              </div>
            </v-card>
          </v-col>
        </v-row>
        <v-row :style="`height: 20%`" dense v-show="showTable">
          <v-col class="px-1">
            <v-card style="height: 100%" class="mx-auto pt-1">
              <div
                :style="`height: 100%; position: relative;`"
                class="white"
              >
                <div style="height: 100%; position: relative" class="white" v-if="initilized">
                  <div style="height: calc(100%); position: relative" class="d-flex justify-center align-end">
                    <div style="height: calc(100%); width: calc(100%); position: relative">
                      <data-table ref="datatable" :display="showTable"/>
                    </div>
                  </div>
                </div>
              </div>
            </v-card>
          </v-col>
        </v-row>
      </v-container>
      <div class="svg-tooltip"
      v-html="tooltipview.content"
      :style="{
        left: `${Math.min(page.width - tooltipview.width, tooltipview.x + 10)}px`,
        top: `${tooltipview.y - 10}px`,
        'max-width': `${tooltipview.width}px`,
        visibility: tooltipview.visibility
      }">
      </div>
    </v-main>
  </v-app>
</template>

<script>
import Matrix from './components/Matrix.vue'
import Feature from './components/Feature.vue'
import Info from './components/Info.vue'
import { mapActions, mapGetters, mapState } from "vuex"
// import SVGTable from './libs/svgtable'
// import * as d3 from 'd3'
import DataTable from './components/DataTable.vue'

export default {
  name: 'App',
  components: {
    Matrix,
    Feature,
    Info,
    DataTable
  },
  computed: {
    ...mapGetters(['model_info', 'rule_info']),
    ...mapState(['tooltipview', 'highlighted_sample', 'data_table', 'crossfilter', 'data_header', 'page', 'matrixview'])
  },
  watch: {
    crossfilter(val) {
      // this.renderDataTable()
    },
    highlighted_sample(val) {
      // this.renderDataTable()
    }
  },
  methods: {
    ...mapActions(['fetchRawdata', 'updateMatrixLayout', 'updatePageSize', 'setReady']),
    toggleDatatable () {
      this.showTable = !this.showTable
      this.$nextTick(() => {
        this.$refs.matrix?.onResize()
        this.$refs.datatable?.onResize()
      })
    },
    onResize(){
      const width = this.$refs.main_container.getBoundingClientRect().width
      const height = this.$refs.main_container.getBoundingClientRect().height
      this.updatePageSize({ width, height })
    },
    renderDataTable() {
      // const width = this.$refs.tableview.parentNode.getBoundingClientRect().width
      // const height = 250

      // const svg = d3.select(this.$refs.tableview)
      //     .attr("width", width)
      //     .attr("height", height);

      // svg.selectAll('*').remove()

      // const reordered_data = 
      //   this.highlighted_sample ? (
      //     this.filtered_data.filter(d => this.highlighted_sample == d._id).concat(
      //       this.filtered_data.filter(d => this.highlighted_sample != d._id)
      //     )
      //   ) : this.filtered_data

      // new SVGTable(svg)
      //     .size([width, height])
      //     .fixedRows(this.highlighted_sample ? 1 : 0)
      //     .fixedColumns(1)
      //     .rowsPerPage(25)    
      //     .defaultNumberFormat(",.0d")
      //     .style({ border: false })
      //     .data(reordered_data)  
      //     .onclick((ctx, cell) => {
      //       const sample_id = this.filtered_data[cell.rowIndex]._id
      //       this.highlightSample(sample_id)
      //     })
      //     //.onhighlight((ctx, d) => { console.log(ctx, ) })
      //     .render();
    }
  },
  data: () => {
    return {
      drawer: null,
      tab: 'Data Table',
      initilized: false,
      showTable: true
    }
  },
  async mounted() {
    await this.fetchRawdata()
    await this.setReady()
    await this.updateMatrixLayout()
    window.addEventListener('resize', this.onResize, { passive: true })
    this.onResize()
    // this.renderDataTable()
    this.initilized = true
  }
}
/*
离散值 留出空隙 不要连续
最下面加一条横线

Not 空心 / Yes 实心
sync with MingYao

crossfilter
规则filter，有个明确的confirm button
histogram debug
*/
</script>

<style lang="scss">
#app {
  font-family: Roboto, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  background: #f7f7f7;
  height: 100%;
  font-size: 16px;
  overflow-y: hidden;
}

.svg-tooltip {
  font-family: -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple   Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
  background: rgba(255, 255, 255, .95);
  border-radius: .1rem;
  border: 1.5px solid #333;
  color: #333;
  font-size: 16px;
  word-wrap:break-word;
  padding: .4rem .6rem;
  position: absolute;
  z-index: 300;
}

.tree-subtitle {
  font-family: Roboto;
  font-size: 20px;
  line-height: 1rem;
  letter-spacing: 0.0125em;
  font-weight: 500;
  fill: rgba(0, 0, 0, 0.6);
  color: rgba(0, 0, 0, 0.6);
}

.tree-text {
  font-family: Arial;
  font-size: 16px;
  line-height: 1.25rem;
  letter-spacing: 0.0125em;
  color: rgba(0, 0, 0, 0.8);
}

@media (min-width: 2560px) {
  .container {
      max-width: 2560px!important;
  }
}

</style>
