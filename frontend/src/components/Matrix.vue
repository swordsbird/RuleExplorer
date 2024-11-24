<template>
  <div ref="matrix_parent" :style="`position: absolute; ${positioning}`" v-resize="onResize">
    <!--div class="text-center" v-if="matrixview.order_keys.length > 0">
      Order by:
      <v-btn
        v-for="item in matrixview.order_keys"
        :key="item.key"
        class="ma-2 pa-1"
        close
        @click:close="chip1 = false"
        style="text-transform: none!important"
      >
        {{ item.name }} {{ item.order == 1 ? '' : '(Descending)'}}
      </v-btn>
    </div-->
    <svg class="matrixdiagram" style="width: 100%; height: 100%">
      <defs>
        <path
          id='return'
          fill = '#555'
          transform='scale(0.025)'
          d = 'M180.864 277.5808c-16.25088 12.288-16.25088 36.7104 0 48.9984L321.9456 433.31072c20.23424 15.3088 49.2544 0.87552 49.2544-24.4992V343.04h201.77408c118.48192 0 214.53312 89.3952 214.53312 199.68 0 110.27968-96.0512 199.68-214.53312 199.68H240.64a30.72 30.72 0 0 0-30.72 30.72v20.48a30.72 30.72 0 0 0 30.72 30.72h332.33408C740.06528 824.32 875.52 698.24512 875.52 542.72s-135.45472-281.6-302.54592-281.6H371.2V195.34848c0-25.37472-29.02016-39.808-49.2544-24.4992L180.864 277.5808z'
        ></path>
      </defs>
      <clipPath id="rule_clip">
        <rect :width="`${matrixview.width + 5}`"
          :height="`${matrixview.height - matrixview.margin.bottom - matrixview.margin.top + 5}`">
        </rect>
      </clipPath>
      <g class="rule_canvas_container"
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width},${matrixview.margin.top})`">
        <g class="rule_outer" clip-path="url(#rule_clip)">
          <g class="rule_canvas" :transform="`translate(0, ${current_scroll})`">
          </g>
        </g>
      </g>
      <g class="header_container"
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width},${matrixview.margin.top})`">
        <g class="header_text">
        </g>
      </g>
      <g class="status_container"
        :transform="`translate(${matrixview.margin.left - matrixview.glyph_width}, ${matrixview.height - matrixview.margin.bottom})`">
      </g>
      <g class="scrollbar_container"></g>
    </svg>
  </div>
</template>

<script>
import { mapActions, mapGetters, mapMutations, mapState } from 'vuex'
import * as d3 from 'd3'
import HistogramChart from "../libs/histogramchart";
import Scrollbar from "../libs/scrollbar";
import utils from "../libs/utils"
import * as axios from 'axios'

function formatPower(x) {
  if (x < 0) return '-' + formatPower(-x)
  if (x == 0) return 0
  const e = Math.floor(Math.log10(x));
  return `10${(e + "").replace(/./g, c => "⁰¹²³⁴⁵⁶⁷⁸⁹"[c] || "⁻")}`;
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export default {
  name: 'Matrix',
  props: {
    positioning: {
      default: 'top: 0px; left: 0px; right: 0px; bottom: 0px',
      type: String
    }
  },
  data() {
    return {
      current_col: null,
      current_row: null,
      current_scroll: 0,
    }
  },
  computed: {
    ...mapState(['server_url', 'dataset', 'highlighted_sample', 'data_table', 'data_features', 'matrixview', 'featureview', 'layout', 'primary', 'glyph', 'session_id']),
    ...mapGetters(['model_info', 'rule_info', 'view_info']),
  },
  watch: {
    layout(val) {
      if (val != null) {
        this.render()
      }
    },
    highlighted_sample(val) {
      this.render()
    }
  },
  // beforeDestroy () {
  //   if (typeof window === 'undefined') return
  //   window.removeEventListener('resize', this.onResize, { passive: true })
  // },
  // async mounted() {
  //   window.addEventListener('resize', this.onResize, { passive: true })
  //   this.onResize()
  // },
  methods: {
    ...mapActions(['getDistribution', 'updateCurrentRule', 'tooltip', 'orderColumn', 'orderRow', 'showRules', 'updateMatrixSize', 'updatePage', 'updateFeaturePosition', 'updateFeaturePin']),
    onResize() {
      const width = this.$refs.matrix_parent.getBoundingClientRect().width
      const height = this.$refs.matrix_parent.getBoundingClientRect().height
      this.updateMatrixSize({ width, height })
    },
    render() {
      const self = this
      // const min_confidence = 5
      const matrixview = this.matrixview
      const header_offset = { x: 100, y: 8 } //this.primary.has_primary_key ? 20 : 5 }

      const svg = d3.select(".matrixdiagram")
        .attr('width', this.matrixview.width)
        .attr('height', this.matrixview.height)

      /*<svg style="width:24px;height:24px" viewBox="0 0 24 24">
    <path fill="currentColor" d="M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z" />
</svg>*/
      const fixed_element = svg.selectAll('.fixed')
        .data(['fixed'])
        .enter().append('g')
        .attr('class', 'fixed')

      fixed_element
        .selectAll('symbol')
        .data(Object.keys(self.glyph))
        .enter()
        .append("symbol")
        .attr("id", d => d)
        .attr("viewBox", "0 0 24 24")
        .append("path")
        .attr("fill", d => self.glyph[d].fill)
        .attr("d", d => self.glyph[d].path)

      fixed_element
        .append("text")
        .attr('class', 'tree-subtitle')
        .attr("dx", 10)
        .attr("dy", 28)
        .text('Rules')

      const view_height = this.matrixview.height - this.matrixview.margin.bottom - this.matrixview.margin.top
      const view_width = this.matrixview.width

      const scroll = svg.select('.scrollbar_container')
      scroll.selectAll('*').remove()

      if (0 && view_height < this.layout.height) {
        this.current_scroll = 5
        const barheight = view_height * view_height / (this.layout.height + 50)
        new Scrollbar(scroll)
          .vertical(true)
          .sliderLength(barheight)
          .position(view_width - 20, this.matrixview.margin.top, view_height)
          .onscroll((x, sx, delta) => this.current_scroll = 5 - sx / barheight * view_height)
          .attach()
      } else {
        this.current_scroll = 0
      }

      const header_container = svg.select(".header_container")
      const rule_canvas = svg.select(".rule_canvas")
      const status_container = svg.select(".status_container")

      const layout = this.layout

      function brushed({ selection }) {
        //rule_canvas.selectAll('g.row')
        //  .select(".glyph").select("circle")
        //  .attr("fill", "darkgray")
        /*
        if (self.matrixview.zoom_level > 0) {
          self.showRepresentRules()
        } else {
        */
        const selected_rules = layout.rows
          .filter(d => d.y >= selection[0] && d.y + d.height <= selection[1])
          .map(d => d.rule.id)
        self.showRules(selected_rules)
        //}
      }

      function brushing({ selection }) {
        const selected_rules = layout.rows
          .filter(d => d.y >= selection[0] && d.y + d.height <= selection[1])
          .map(d => d.rule.id)
        const is_selected_rule = new Set(selected_rules)
        rule_canvas.select('rect.selection').attr('width', matrixview.width)
        rule_canvas.selectAll('g.row')
          .select(".glyph circle")
          .attr("fill", d => is_selected_rule.has(d.rule.id) ? "#333" : "darkgray")
        rule_canvas.selectAll('g.row')
          .select(".glyph line")
          .attr("stroke", d => is_selected_rule.has(d.rule.id) ? "#333" : "darkgray")
        rule_canvas.selectAll('g.row')
          .select(".glyph rect")
          .attr("stroke", d => is_selected_rule.has(d.rule.id) ? "#333" : "darkgray")
      }

      const xrange = [Math.min(...layout.cols.map(d => d.x)), Math.max(...layout.cols.map(d => d.x)) + matrixview.coverage_width]
      const yrange = [Math.min(...layout.rows.map(d => d.y)), Math.max(...layout.rows.map(d => d.y + d.height))]

      function updateStatus() {
        status_container.selectAll('*').remove()
        if (self.layout.fold_info.fold) {
          const left_button = status_container.append('g')
            .attr('transform', `translate(${layout.fold_info.left_x}, 35)`)

          if (layout.fold_info.page) {
            left_button.append('use')
              .attr('href', layout.fold_info.page == 1 ? '#1dot' : '#3dots')
              .attr('x', -25)
              .attr('y', -8)
              .attr('width', 36)
              .attr('height', 36)
              .attr('opacity', .5)

            left_button.append('use')
              .attr('href', '#prev-page')
              .attr('x', -25 - 10 - (layout.fold_info.page > 1 ? 10 : 0))
              .attr('y', -8)
              .attr('width', 36)
              .attr('height', 36)
              .attr('opacity', .5)
              .on('mouseover', function () {
                if (!layout.fold_info.has_left_page) return
                d3.select(this).attr("opacity", 1)
              }).on('mouseout', function () {
                if (!layout.fold_info.has_left_page) return
                d3.select(this).attr("opacity", .5)
              }).on('click', function (ev, d) {
                if (!layout.fold_info.has_left_page) return
                self.updatePage(-1)
              })
          }

          const right_button = status_container.append('g')
            .attr('transform', `translate(${layout.fold_info.right_x}, 35)`)

          if (layout.fold_info.next) {
            right_button.append('use')
              .attr('href', layout.fold_info.next == 1 ? '#1dot' : '#3dots')
              .attr('x', -11)
              .attr('y', -8)
              .attr('width', 36)
              .attr('height', 36)
              .attr('opacity', .5)

            right_button.append('use')
              .attr('href', '#next-page')
              .attr('x', -11 + 10 + (layout.fold_info.next > 1 ? 10 : 0))
              .attr('y', -8)
              .attr('width', 36)
              .attr('height', 36)
              .attr('opacity', .5)
              .on('mouseover', function () {
                if (!layout.fold_info.has_right_page) return
                d3.select(this).attr("opacity", 1)
              }).on('mouseout', function () {
                if (!layout.fold_info.has_right_page) return
                d3.select(this).attr("opacity", .5)
              }).on('click', function (ev, d) {
                if (!layout.fold_info.has_right_page) return
                self.updatePage(1)
              })
          }
        }

        const back_btn = status_container.append('g')
          .attr('transform', `translate(0, ${-layout.height - 20})`)
          .on('mouseover', function () {
            if (self.matrixview.zoom_level == 0) return
            d3.select(this).select('rect.background').attr('fill-opacity', .8).attr('stroke-width', 1)
              .attr('stroke', 'black')
          }).on('mouseout', function () {
            if (self.matrixview.zoom_level == 0) return
            d3.select(this).select('rect.background').attr('fill-opacity', .3).attr('stroke-width', .3)
              .attr('stroke', 'lightgray')
          }).on('click', function (ev, d) {
            if (self.matrixview.zoom_level == 0) return
            self.showRules(null)
          })

        back_btn.append('rect')
          .attr('class', 'background')
          .attr('y', -3)
          .attr('x', 0)
          .attr('width', 73)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)

        back_btn.append('text')
          .attr('dx', 4)
          .attr('dy', 13)
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .attr('opacity', self.matrixview.zoom_level == 0 ? 0.5 : 1)
          .text('return')

        back_btn
          .append('use')
          .attr('href', '#return')
          .attr('x', 45)
          .attr('y', -6)
          .attr('opacity', self.matrixview.zoom_level == 0 ? 0.5 : 1)
          .attr('width', self.matrixview.font_size + 4)
          .attr('height', self.matrixview.font_size + 4)

        const count_btn = status_container.append('g')
          .attr('transform', `translate(0,35)`)
          .on('mouseover', function () {
            d3.select(this).select('rect.background').attr('fill-opacity', .8).attr('stroke-width', 1)
              .attr('stroke', 'black')
          }).on('mouseout', function () {
            d3.select(this).select('rect.background').attr('fill-opacity', .3).attr('stroke-width', .3)
              .attr('stroke', 'lightgray')
          }).on('click', function (ev, d) {
            self.orderRow()
          })

        count_btn.append('rect')
          .attr('class', 'background')
          .attr('y', -3)
          .attr('x', 0)
          .attr('width', 98)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)

        if (layout.has_pin) {
          status_container.append('rect')
            .attr('class', 'background')
            .attr('y', 4)
            .attr('height', 50)
            .attr('stroke', matrixview.cell.stroke_color)
            .attr('stroke-width', matrixview.cell.stroke_width)
            .attr('fill', 'none')
            .attr('x', layout.main_start_x)
            .attr('width', layout.fold_info.pin_width)

          status_container.append('rect')
            .attr('class', 'background')
            .attr('y', 4)
            .attr('height', 50)
            .attr('stroke', matrixview.cell.stroke_color)
            .attr('stroke-width', matrixview.cell.stroke_width)
            .attr('fill', 'none')
            .attr('x', layout.fold_info.left_x)
            .attr('width', layout.fold_info.right_x - layout.fold_info.left_x)
        } else {
          status_container.append('rect')
            .attr('class', 'background')
            .attr('y', 4)
            .attr('height', 50)
            .attr('stroke', matrixview.cell.stroke_color)
            .attr('stroke-width', matrixview.cell.stroke_width)
            .attr('fill', 'none')
            .attr('x', layout.main_start_x)
            .attr('width', layout.main_width)
        }
        status_container.append('line')
          .attr('class', 'background')
          .attr('y1', 34)
          .attr('y2', 34)
          .attr('stroke', matrixview.cell.stroke_color)
          .attr('stroke-width', matrixview.cell.stroke_width)
          .attr('fill', 'none')
          .attr('x1', layout.main_start_x)
          .attr('x2', layout.main_start_x + layout.main_width)
        /*
                count_btn.append('path')
                  .attr('d', "M41 288h238c21.4 0 32.1 25.9 17 41L177 448c-9.4 9.4-24.6 9.4-33.9 0L24 329c-15.1-15.1-4.4-41 17-41zm255-105L177 64c-9.4-9.4-24.6-9.4-33.9 0L24 183c-15.1 15.1-4.4 41 17 41h238c21.4 0 32.1-25.9 17-41z")
                  .attr('transform', "translate(84.5, 3) scale(0.04) rotate(90)")
                  .attr("opacity", matrixview.sort_by_cover_num ? .8 : .2)
        */
        count_btn.append('text')
          .attr('dx', 4)
          .attr('dy', 13)
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .text('importance')

        const status_orders = status_container.select('g.order')
          .data(self.matrixview.order_keys).enter()
          .append('g')
          .attr('class', 'order')
          .attr('transform', (d, i) => `translate(${135 + i * 110}, 60)`)

        status_orders.append('rect')
          .attr('width', 100)
          .attr('height', 20)
          .attr('stroke', 'lightgray')
          .attr('stroke-width', .3)
          .attr('fill', 'lightgray')
          .attr('fill-opacity', .3)
      }

      function updateCols() {
        const cols_data = layout.cols
        let related_cells
        let start_x = 0, prev = ''
        function dragstarted(event, d) {
          d3.select(this).raise()
          d3.select(this).select(".header").attr("stroke", "black")
          d3.select(this).select(".background").attr("stroke", "black")
          related_cells = d3.selectAll(".cell").filter(e => e.name == d.name)
          related_cells.attr('transform', 'translate(0, 0)')
          start_x = event.x

          let left = 0
          header_container.selectAll('g.col')
            .each((e, i) => {
              if (e.x < d.x && e.x > left) {
                left = e.x
                prev = e.name
              }
            })
        }

        function dragged(event, d) {
          d3.select(this).attr("transform", `translate(${event.x},${d.y})`)
          related_cells.attr('transform', `translate(${event.x - start_x}, 0)`)
        }

        function dragended(event, d) {
          d3.select(this)
            .select('rect.background')
            .attr('stroke', matrixview.cell.stroke_color)
            .attr('stroke-width', matrixview.cell.stroke_width)
          d3.select(this)
            .select('path.header')
            .attr('fill-opacity', matrixview.cell.header_opacity)
            .attr('stroke', matrixview.cell.stroke_color)
            .attr('stroke-width', matrixview.cell.stroke_width)

          if (!d.pin && event.x < layout.fold_info.left_x + 10 ||
            d.pin && event.x > layout.fold_info.left_x + 50) {
            if (Math.abs(event.x - start_x) > 50) {
              related_cells
                .attr('transform', `translate(0, 0)`)
              self.updateFeaturePin(d.name)
            } else {
              related_cells
                .transition().duration(matrixview.duration)
                .attr('transform', `translate(0, 0)`)
            }
          } else {
            let left = 0, target = ''
            header_container.selectAll('g.col')
              .each((e, i) => {
                if (e.x < event.x && e.x > left) {
                  left = e.x
                  target = e.name
                }
              })
            if (prev == target || d.name == target) {
              related_cells
                .transition().duration(matrixview.duration)
                .attr('transform', `translate(0, 0)`)
              d3.select(this)
                .transition().duration(matrixview.duration)
                .attr("transform", `translate(${start_x},${d.y})`)
            } else {
              console.log('move', d.name, target)
              related_cells
              .transition().duration(matrixview.duration)
                .attr('transform', `translate(0, 0)`)
              self.updateFeaturePosition({ name: d.name, target })
            }
          }
        }

        const drag = d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended)

        /*
        const topleft_text = header_container.select('.header_text')
        topleft_text.append('text')
          .attr('dx', -5)
          .attr('font-size', '16px')
          .attr('font-family', 'Arial')
          .attr('dy', -60)
          .style('text-anchor', 'start')
          .text(self.view_info)
        */
        let col = header_container.selectAll('g.col')
          .data(cols_data, d => d.index)

        col.exit().selectAll('*').remove()

        let col_join = col.enter().append('g')
          .attr('class', 'col')
          .attr('transform', d => `translate(${d.x},${d.y})`)

        col = header_container.selectAll('g.col')
          .data(cols_data, d => d.index)
          .style("display", d => d.show ? "block" : "none")
          .call(drag)

        col_join.append('path').attr('class', 'header')
        col_join.append('text').attr('class', 'label')
        col_join.append('text').attr('class', 'desc')
        col_join.append('g').attr('class', 'axis')
        col_join.append('text').attr('class', 'count')

        const col_background = col_join
          .append('g').attr('class', 'col-content')

        col_background.append('rect').attr('class', 'background')

        function headerInteraction(x) {
          x.on('mouseover', function () {
            d3.select(this)
              .raise()
            d3.select(this)
              .select('rect.background')
              .attr('stroke', matrixview.cell.highlight_stroke_color)
              .attr('stroke-width', matrixview.cell.highlight_stroke_width)
            d3.select(this)
              .select('path.header')
              .attr('fill-opacity', matrixview.cell.highlight_header_opacity)
              .attr('stroke', matrixview.cell.highlight_stroke_color)
              .attr('stroke-width', matrixview.cell.highlight_stroke_width)
          }).on('mouseout', function () {
            d3.select(this)
              .select('rect.background')
              .attr('stroke', matrixview.cell.stroke_color)
              .attr('stroke-width', matrixview.cell.stroke_width)
            d3.select(this)
              .select('path.header')
              .attr('fill-opacity', matrixview.cell.header_opacity)
              .attr('stroke', matrixview.cell.stroke_color)
              .attr('stroke-width', matrixview.cell.stroke_width)
          }).on('click', function (ev, d) {
            self.orderColumn(d.index)
          })
        }

        col.select('path.header')
          .attr('fill', 'lightgray')
          .attr('fill-opacity', matrixview.cell.header_opacity)
          .attr('stroke', matrixview.cell.stroke_color)
          .attr('stroke-width', matrixview.cell.stroke_width)

        col.select('path.header')
          .transition().duration(matrixview.duration)
          .attr('d', d => {
            return `M0,0 L${d.width},0 L${d.width},${-header_offset.y} L${d.width + header_offset.x * 1.5},${-header_offset.y - header_offset.x}
                         L${header_offset.x * 1.5},${-header_offset.y - header_offset.x} L${0},${-header_offset.y} z`
          })

        col.selectAll('g.bottom-hint').remove()
        const bottom_glyph = col.filter(d => d.hint_change)
          .append('g')
          .attr('class', 'bottom-hint')
          .attr('transform', d => `translate(${25},${d.height + 4})`)

        bottom_glyph
          .append('use')
          .attr('href', d => `#${d.hint_change}`)
          .attr('width', self.matrixview.font_size + 4)
          .attr('height', self.matrixview.font_size + 4)

        col.selectAll('g.top-hint').remove()
        const top_glyph = col.filter(d => d.pin)
          .append('g')
          .attr('class', 'top-hint')
          .attr('transform', d => `translate(${50 + header_offset.x}, ${-header_offset.x - 30})`)

        top_glyph
          .append('use')
          .attr('href', '#pin')
          .attr('width', self.matrixview.font_size + 4)
          .attr('height', self.matrixview.font_size + 4)

        col.selectAll('.category').remove()

        col.filter(d => d.show_axis && d.type == 'category')
          .each(function (d) {
            const width = d.width / d.values.length
            const offset_x = header_offset.x - 20
            const offset_y = header_offset.y
            let category = d3.select(this)
              .selectAll('.category')
              .data(d.values).enter()
              .append('g')
              .attr('class', 'category')
              .attr('transform', (e, i) => `translate(${width * i}, 0)`)

            category
              .append('path')
              .attr('fill', 'lightgray')
              .attr('fill-opacity', matrixview.cell.header_opacity)
              .attr('stroke', matrixview.cell.stroke_color)
              .attr('stroke-width', matrixview.cell.stroke_width)
              .attr('d', d => {
                return `M0,0 L${width},0 L${width},${-offset_y} L${width + offset_x * 1.5},${-offset_y - offset_x}
                            L${offset_x * 1.5},${-offset_y - offset_x} L${0},${-offset_y} z`
              })

            const dy = width / 2 - header_offset.y + 1.5
            const len = Math.min(matrixview.maxlen, Math.max(...d.values.map(e => e.length)))
            category
              .append('text')
              .attr('transform', `rotate(-35)`)
              .attr('font-size', `${self.matrixview.font_size - 2}px`)
              .attr('font-family', 'Arial')
              .attr('dx', dy + 16 + (matrixview.maxlen - len) / 2 * 3)
              .attr('dy', dy)// header_offset.y)
              .text(name => name.length < matrixview.maxlen ? name : name.slice(0, matrixview.maxlen) + "...")
            //.style('user-select', 'none')
          })

        col.call(headerInteraction)

        function IsAlpha(cCheck) {
          return ((('a' <= cCheck) && (cCheck <= 'z')) || (('A' <= cCheck) && (cCheck <= 'Z')))
        }

        col.select('text.label')
          .attr('transform', d => d.show_axis && d.type == 'category' ? '' : `rotate(-35)`)
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .attr('dx', d => d.show_axis ? (40 + (d.type == 'category' ? header_offset.x : 0)) : 30)
          .attr('dy', d => d.show_axis && d.type == 'category' ? -header_offset.x + 5 : 16 - header_offset.y)
          .text(d => {
            let prefix = ''
            for (let i = 0; i < self.matrixview.order_keys.length; ++i) {
              if (self.matrixview.order_keys[i].key == d.index) {
                prefix = self.matrixview.order_keys[i].order == 1 ? '▲' : '▼'
              }
            }
            let maxlen = utils.naturallyLength(d.name, matrixview.maxlen)
            let name = d.name.slice(0, maxlen)
            if (d.display_name) {
              name = d.display_name
            }
            name = name.length < maxlen ? name : name.slice(0, maxlen) + (d.show_axis ? "" : "...")
            return prefix + name
          })
          .style('user-select', 'element')

        col.select('text.desc')
          .attr('transform', `rotate(-35)`)
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .attr('display', d => d.show_axis ? 'block' : 'none')
          .attr('dx', 65)
          .attr('dy', d => {
            return 16 - header_offset.y + self.matrixview.font_size
          })
          .text(d => {
            let maxlen = utils.naturallyLength(d.name, matrixview.maxlen)
            let name = d.name
            if (d.display_name) {
              name = d.display_name
            }
            name = name.length < maxlen ? '' : name.slice(maxlen)
            if (name.length > matrixview.maxlen + 2) {
              name = name.slice(0, matrixview.maxlen) + '...'
            }
            return name
          })
          .style('user-select', 'element')
        /*
        col.select('text.label')
          .attr('transform', d => d.show_axis ? '' : `rotate(-35)`)
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .attr('dx', d => {
            if (!d.show_axis) {
              return 30
            } else if (d.type == 'category') {
              return 40 + header_offset.x
            } else {
              return 40
            }
          })
          .attr('dy', d => {
            if (!d.show_axis) {
              return 16 - header_offset.y
            } else if (d.type == 'category') {
              return -header_offset.x + 5
            } else {
              return -header_offset.y - 13
            }
          })
          .text(d => {
            let prefix = ''
            for (let i = 0; i < self.matrixview.order_keys.length; ++i) {
              if (self.matrixview.order_keys[i].key == d.index) {
                prefix = self.matrixview.order_keys[i].order == 1 ? '▲' : '▼'
              }
            }
            const maxlen = d.show_axis ? matrixview.maxlen + 5 : matrixview.maxlen
            let name = d.name
            if (d.display_name) {
              name = d.display_name
            }
            name = name.length < maxlen ? name : name.slice(0, maxlen) + "..."
            return prefix + name
          })
          .style('user-select', 'element')

        col.select('text.desc')
          .attr('transform', `rotate(-35)`)
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .attr('display', 'none')//d => d.show_axis || (!d.type) ? 'none' : 'block')
          .attr('dx', 36)
          .attr('dy', d => {
            return 16 - header_offset.y + self.matrixview.font_size
          })
          .attr('fill', 'darkgray')
          .text(d => {
            if (d.type == 'number') {
              return `${d.range[0]}~${d.range[1]}`
            } else if (d.type == 'category') {
              if (d.name == 'PriorDefault') {
                return 'Default, Good standing'
              } else if (d.name == 'Job') {
                return '#0, #1, #2, ..., #13'
              } else if (d.name == 'BankCustomer') {
                return 'No, Yes'
              } else if (d.name == 'Citizen') {
                return 'Birth, Other, Temp'
              } 
              return d.values.join(',').slice(0, 25)
            }
          })
          .style('user-select', 'element')
        */

        col.select('rect.background')
          .attr('stroke', matrixview.cell.stroke_color)
          .attr('stroke-width', matrixview.cell.stroke_width)
          .attr('height', d => d.height + 34)
          .attr('fill', 'none')

        col.select('rect.background')
          .transition().duration(matrixview.duration)
          .attr('width', d => d.width)

        col.select('text.count')
          .attr('font-size', `${self.matrixview.font_size}px`)
          .attr('font-family', 'Arial')
          .attr('text-anchor', 'middle')
          .attr('dx', d => d.width / 2)
          .attr('dy', d => d.height + 50)
          .text(d => d.count > 0 ? d.count : '')

        col.selectAll('.bar').remove()

        col.selectAll('.bar')
          .data(d => d.hist || []).enter()
          .append('rect')
          .attr('class', 'bar')
          .attr('height', d => d.height)
          .attr('width', d => d.width)
          .attr('x', d => d.x)
          .attr('y', d => d.y)
          .attr('fill', d => d.fill)

        col.filter(d => !d.show_axis && d.type == 'number').each(function (d) {
          d3.select(this).select('g.axis').selectAll('*').remove()
        })
        col.filter(d => d.show_axis && d.type == 'number').each(function (d) {
          if (d.range[1] > 1e7) {
            d3.select(this).select('g.axis')
              //.attr('transform', `translate(0,${-header_offset.y + 5})`)
              .call(d3.axisTop(d.scale.scale()).ticks(2)
                .tickFormat(formatPower))
          } else {
            d3.select(this).select('g.axis')
              //.attr('transform', `translate(0,${-header_offset.y + 5})`)
              .call(d3.axisTop(d.scale.scale()).ticks(4))//, "~s"))
          }
          if (d.range[1] > 1000) {
            const ts = d3.select(this).select('g.axis').selectAll('.tick').select('text')
            for (let i = 1; i < ts._groups[0].length; i += 2) {
              d3.select(ts._groups[0][i]).remove()
            }
          }
          d3.select(this).select('g.axis').raise()
        })

        col
          .transition().duration(matrixview.duration)
          .attr('transform', d => `translate(${d.x},${d.y})`)
      }

      function updateRows() {
        rule_canvas.selectAll("g.brush").remove()
        let canvas_brush = rule_canvas
          .append('g').attr('class', 'brush')

        // console.log(layout.rows.map(d => d.rule.represent))
        let row = rule_canvas.selectAll('g.row')
          .data(layout.rows, d => d.rule.id)

        row.exit()
          .style('opacity', 0).remove()

        let row_join = row.enter().append('g')
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .attr('class', 'row')
          .style('opacity', 0)

        row_join.append('rect')
          .attr('class', 'background')

        row_join.selectAll(".glyph")
          .data(d => [d]).enter()
          .append('g')
          .attr('class', 'glyph')
          .attr('transform', `translate(2.5, 0)`)

        row_join.selectAll(".extend-glyph")
          .data(d => [d]).enter()
          .append('g')
          .attr('class', 'extend-glyph')
          .attr('transform', d => `translate(${Math.max(...d.extends.map(e => e.x1 + e.width))}, 0)`)

        row = row.merge(row_join)

        row.select('rect.background')
          .transition().duration(matrixview.duration)
          .attr('x', 1)
          .attr('y', -1)
          .attr('width', d => d.width)
          .attr('height', d => d.height + 2)
          //.attr('fill', d => !d.is_selected ? 'lightgray' : 'yellow')
          //.attr('opacity', d => d.is_selected ? 0.5 : 0)
          //.attr('fill', d => !d.is_selected ? 'lightgray' : 'gray')
          .attr('fill-opacity', 0)
          .attr('fill', 'white')
          .attr('stroke', 'black')
          .attr('stroke-width', d => d.is_selected ? 2 : 0)
        //          .attr('opacity', d => d.is_selected ? 0.5 : 0)

        row.on('mouseenter', function (ev, d) {
          d3.select(this).select('.background')
            .attr('stroke-width', d => (d.is_selected ? 2 : 0) + 1)
        }).on('mouseout', function (ev, d) {
          d3.select(this).select('.background')
            .attr('stroke-width', d => d.is_selected ? 2 : 0)
        }).on('click', function (ev, d) {
          console.log('current rule', d)
          self.updateCurrentRule(d)
        })

        row.select('.extend-glyph').selectAll('*').remove()
        let extend_glyph = row
          .select('.extend-glyph')
        extend_glyph.selectAll('*').remove()

        extend_glyph.filter(d => d.hint_change != '')
          .append('use')
          .attr('href', d => `#${d.hint_change}`)
          .attr('x', 4)
          .attr('y', d => -d.height / 2 - 2)
          .attr('width', self.matrixview.font_size)
          .attr('height', self.matrixview.font_size)

        row.select('.glyph').selectAll('*').remove()
        let represent_glyph = row//.filter(d => d.rule.represent)
          .select('.glyph')
        //.attr("opacity", .5)

        represent_glyph.selectAll('*').remove()

        represent_glyph
          .append('line')
          .attr('class', 'extend')
          .attr('x1', d => d.rule.level * matrixview.indent_size)
          .attr('x2', 90)
          .attr('y1', d => d.glyphheight / 2 + 1)
          .attr('y2', d => d.glyphheight / 2 + 1)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')

        represent_glyph
          .append('circle')
          .attr('class', 'extend')
          .attr('cx', 90)
          .attr('cy', d => d.glyphheight / 2 + 1)
          .attr('r', d => d.glyphheight / 2 - 0.5)
          .attr('fill', 'darkgray')
          .attr('stroke', 'none')

        let rep_glyph = represent_glyph
          .append('g')
          .attr('class', 'represent_glyph')
          .on('mousemove', function (ev, d) {
            if (d.attr.num_children == 0) return
            self.tooltip({ type: "text", data: `represent ${Number(d.attr.num_children)} rules` })
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY } })
          })
          .on('mouseover', function (ev, d) {
            if (d.attr.num_children == 0) return
            self.tooltip({ type: "show" })
            d3.select(this).select("rect.bg")
              .transition().duration(matrixview.duration / 2)
              .attr('x', d => d.rule.level * matrixview.indent_size - 1)
              .attr('y', 1 - 1.5)
              .attr('width', d => d.attr.num + 2)
              .attr('height', d => d.glyphheight + 3)
              .attr('stroke', "#333")
              .attr("stroke-width", '2.5px')
            d3.select(this).select(".arrow")
              .style("display", "block")
            d3.select(this.parentNode.parentNode).raise()
            /*
            d3.select(this).selectAll("rect.dot")
              .transition().duration(matrixview.duration / 2)
              .attr('width', 3)
              .attr('height', 3)
              .attr('fill', "#333")*/
            d3.select(this.parentNode).select("line.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('stroke', '#333')
              .attr('stroke-width', '3px')
            d3.select(this.parentNode).select("circle.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('r', d => d.glyphheight / 2)
              .attr('fill', '#333')
          })
          .on('mouseout', function (ev, d) {
            if (d.attr.num_children == 0) return
            self.tooltip({ type: "hide" })
            d3.select(this).select("rect.bg")
              .transition().duration(matrixview.duration / 2)
              .attr('x', d => d.rule.level * matrixview.indent_size)
              .attr('y', 1)
              .attr('width', d => d.attr.num)
              .attr('height', d => d.glyphheight)
              .attr('stroke', "darkgray")
              .attr("stroke-width", '1.5px')
            d3.select(this).select(".arrow")
              .style("display", d => (self.matrixview.zoom_level > 0 && d.rule.level < self.matrixview.zoom_level) ? "block" : "none")
            /*
          d3.select(this).selectAll("rect.dot")
            .transition().duration(matrixview.duration / 2)
            .attr('width', 2)
            .attr('height', 2)
            .attr('fill', "darkgray")*/
            d3.select(this.parentNode).select("line.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('stroke', 'darkgray')
              .attr('stroke-width', '2px')
            d3.select(this.parentNode).select("circle.extend")
              .transition().duration(matrixview.duration / 2)
              .attr('r', d => d.glyphheight / 2 - 0.5)
              .attr('fill', 'darkgray')
          })
          .on('click', function (ev, d) {
            if (self.matrixview.zoom_level > d.rule.level) {
              self.showRules(null)
            } else {
              self.showRules([d.rule.id])
            }
          })

        rep_glyph
          .append('rect')
          .attr('class', 'bg')
          .attr('x', d => d.rule.level * matrixview.indent_size)
          .attr('y', 1)
          .attr('width', d => d.attr.num)
          .attr('height', d => d.glyphheight)
          .attr('fill', '#f7f7f7')
          .attr('stroke', 'darkgray')
          .style("display", d => "block")//(d.attr.num_children > 0 || d.rule.level == 0) ? "block" : "none")
          .attr('stroke-width', '1.5px')

        rep_glyph
          .append('use')
          .attr('href', d => self.matrixview.zoom_level == d.rule.level ? "#marker-expand" : "#markercollapse")
          .attr("x", d => d.rule.represent ? -3 : 27)
          .attr('class', "arrow")
          .attr("width", 15)
          .attr("height", 15)
          .attr("y", -3)
          .style("display", d => (self.matrixview.zoom_level > 0 && d.rule.level < self.matrixview.zoom_level) ? "block" : "none")
          .style("fill", "#333")
        /*
                let represent_glyph_dot = rep_glyph
                  .append("g")
                  .attr('class', "dot")
        
                represent_glyph_dot
                  .append("rect")
                  .attr("x", 12)
                  .attr("y", d => d.height / 2 - 1)
                  .attr("height", 2)
                  .attr("width", 2)
                  .attr('class', "dot")
                  .attr("fill", "darkgray")
        
                represent_glyph_dot
                  .append("rect")
                  .attr("x", 16)
                  .attr("y", d => d.height / 2 - 1)
                  .attr("height", 2)
                  .attr("width", 2)
                  .attr('class', "dot")
                  .attr("fill", "darkgray")
        
                represent_glyph_dot
                  .append("rect")
                  .attr("x", 20)
                  .attr("y", d => d.height / 2 - 1)
                  .attr("height", 2)
                  .attr("width", 2)
                  .attr('class', "dot")
                  .attr("fill", "darkgray")
        */
        let nonrepresent_glyph = row.filter(d => !d.rule.represent)
          .select('.glyph')
        //.attr("opacity", .5)

        /*
        nonrepresent_glyph
          .append('line')
          .attr('x1', 30)
          .attr('x2', 30)
          .attr('y1', d => d.glyphheight / 2 - d.lastheight)
          .attr('y2', d => d.glyphheight / 2)
          .attr('stroke', 'darkgray')
          .attr('stroke-width', '2px')
          .attr('class', 'parent')

        nonrepresent_glyph
          .select('line.parent')
          .attr('opacity', 0)
        */

        nonrepresent_glyph
          .select('line.parent')
          .transition().duration(matrixview.duration)
          .delay(matrixview.duration)
          .attr('opacity', 1)

        /*
                nonrepresent_glyph
                  .append('rect')
                  .attr('class', 'bg')
                  .attr('fill', '#f7f7f7')
                  .attr('stroke', 'darkgray')
                  .attr('stroke-width', '1.5px')
        */
        row_join = row_join.merge(row)

        const brush = d3.brush()
          .on("end", brushed)

        canvas_brush
          .call(
            d3.brushY().extent([[xrange[0] - 30, yrange[0]], [xrange[0], yrange[1]]])
              .on("brush", brushing)
              .on("end", brushed)
          )

        let row_extend = row_join.selectAll('g.extended')
          .data(d => d.extends)
          .enter().append('g')
          .attr('class', 'extended')
          .attr('opacity', 1)
        row_extend.exit().selectAll('*').remove()

        let cell = row_join.selectAll('g.cell')
          .data(d => d.items)
          .enter().append('g')
          .attr('class', 'cell')
          .attr('opacity', 1)

        row_extend.append('rect').attr('class', 'bar')
        cell.append('rect').attr('class', 'barbg')
        cell.append('rect').attr('class', 'glyph')
        cell.append('g').attr('class', 'bargroup')

        row = row.merge(row_join)

        cell = row.selectAll('g.cell')
          .style("display", d => d.show ? "block" : "none")
          .on("mouseover", function () {
            self.tooltip({ type: "show" })
          })
          .on("mouseout", function () {
            self.tooltip({ type: "hide" })
          })
          .on("mousemove", function (ev, d) {
            self.tooltip({
              type: "text",
              data: d.text,
            })
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY } })
          })
          .raise()

        row.selectAll('g.cell')
          .data(d => d.items)
          .exit().remove()

        row_extend = row.selectAll('g.extended')

        cell.select('rect.barbg')
          .style("display", d => (d.is_glyph || d.elements.length == 0 || d.feature.type == 'category') ? "none" : "block")
          .attr('width', d => d.width)
          .attr('height', d => d.height)
          .attr('fill', d => d.bgfill)
          .attr('stroke', d => d.show_hist ? 'darkgray' : 'none')
          .attr('stroke-width', '0.5px')
          .attr('opacity', 1)

        cell.select('rect.barbg')
          .transition().duration(matrixview.duration)
          .attr('x', d => d.x)

        cell.select('rect.glyph')
          .style("display", d => (d.is_glyph && !d.show_hist) ? "block" : "none")
          .attr('width', d => d.width / 2)
          .attr('height', d => d.height)
          .attr('rx', 3)
          .attr('ry', 3)
          .attr('fill', d => d3.interpolateLab('white', d.fill)(0.5))
          .attr('stroke', 'none')
          .attr('stroke-width', '1px')
          .attr('opacity', 1)

        cell.select('rect.glyph')
          .transition().duration(matrixview.duration)
          .attr('x', d => d.x + d.width / 4)

        canvas_brush.raise()

        cell.select(".histogram").remove()

        const cell_bars = cell.select('g.bargroup')
        cell_bars
          .style("display", d => d.is_glyph ? "none" : "block")

        cell_bars
          //.attr("opacity", 0)
          .transition().duration(matrixview.duration)
          .attr('transform', d => `translate(${d.x},${0})`)
        //.attr("opacity", 1)

        cell_bars.selectAll('rect.bar')
          .data(d => d.elements).exit().remove()

        cell_bars.selectAll('rect.bar')
          .data(d => d.elements).enter()
          .append('rect')
          .attr('class', 'bar')

        cell_bars.selectAll('rect.bar')
          .data(d => d.elements)
          .attr('x', d => {
            if (d.show_hist) {
              return d.x0
            } else {
              return d.x0
            }
          })
          .attr('y', 0)
          .attr('width', d => {
            if (d.show_hist) {
              return Math.max(d.x1 - d.x0, 0)
            } else {
              return Math.max(d.x1 - d.x0, 0)
            }
          })
          .attr('height', d => Math.max(d.h, 0))
          .attr('fill', d => {
            if (d.show_hist) {
              return d3.interpolateLab('white', d.fill)(1)
            } else if (!d.neg) {
              return d.fill
            } else {
              return 'white'
            }
          })
          .attr('stroke', d => {
            return 'none'
          })
          .attr('stroke-width', '1.5px')

        const chart_cell = cell.filter(d => d.show_hist)
        setTimeout(() => {
          chart_cell.each(async function (d) {
            let data = await self.getDistribution({
              id: d.id, feature: d.name
            })

            let ticks = d.feature.values

            if (d.feature.type == 'number') {
              data = data.map(e => d.scale(e))
              ticks = [0, d.width]
            }

            const chart = HistogramChart()
              .data(data)
              .valueTicks(ticks)
              .datatype(d.feature.type)
              .width(d.width)
              .height(d.height - 4)
              .color(d.hfill)

            d3.select(this)
              .append("g")
              .attr('class', "histogram")
              .attr("transform", `translate(${d.x}, 4)`)
              .attr("opacity", 0)
              .call(chart)

            d3.select(this)
              .select("g.histogram")
              .transition().duration(matrixview.duration)
              .delay(matrixview.duration)
              .attr("opacity", 1)
          })
        }, 500)

        row_extend.select('rect.bar')
          .on("mouseover", function () {
            self.tooltip({ type: "show" })
          })
          .on("mouseout", function () {
            self.tooltip({ type: "hide" })
          })
          .on("mousemove", function (ev, d) {
            self.tooltip({
              type: "text",
              data: d.text,//`${Number(d.value).toFixed(3)}`
            })
            self.tooltip({ type: "position", data: { x: ev.pageX, y: ev.pageY } })
          })
          .raise()

        row_extend.select('rect.bar')
          .attr('x', d => d.x1)
          .attr('height', d => d.height)
          .attr('fill', d => d.fill)
          .transition().duration(matrixview.duration)
          .attr('width', d => d.x2 - d.x1)

        row
          .transition().duration(matrixview.duration)
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .transition().duration(matrixview.duration)
          .style('opacity', d => self.highlighted_sample === undefined ? 1 : d.samples.has(self.highlighted_sample) ? 1 : 0.3)
        //          .style('opacity', 1)//d => d.rule.represent ? 1 : 0.5)

      }

      function update() {
        updateCols()
        updateRows()
        updateStatus()
      }

      update()
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss"></style>
