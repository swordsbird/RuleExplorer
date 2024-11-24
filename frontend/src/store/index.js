import Vue from 'vue'
import Vuex from 'vuex'
import * as d3 from 'd3'
import * as axios from 'axios'
import state from './state'
import { sortList } from './ordering'

Vue.use(Vuex)

const distribution_cache = {}
let showRuleFlag = 0

function makeScale(type, domain, range) {
  let scale
  if (type == 'linear') {
    scale = d3.scaleLinear()
  } else if (type == 'log') {
    if (domain[0] == 0) {
      domain = [1e-6 + domain[0], domain[1]]
    }
    scale = d3.scaleLog()
  } else if (type == 'sqrt') {
    scale = d3.scaleSqrt()
  }

  scale.domain(domain)
    .range(range)

  function self(x) {
    if (x < domain[0]) {
      x = domain[0]
    } else if (x > domain[1]) {
      x = domain[1]
    }
    return scale(x)
  }
  self.domain = () => domain
  self.range = () => range
  self.scale = () => scale

  return self
}

export default new Vuex.Store({
  state: state,
  mutations: {
    setModelInfo(state, data) {
      state.model_info = data.model_info
      state.session_id = data.session_id
      state.model_info.loaded = true
      state.dataset.label = data.model_info.targets
      const extended_cols = [['Confidence', 'fidelity'], ['Coverage', 'coverage'], ['Anomaly Score', 'anomaly']]
      if (state.model_info.weighted) {
        extended_cols.splice(2, 0, ['Weight', 'weight'])
      }
      state.matrixview.extended_cols = extended_cols.map((d, position) => ({
        position, name: d[0], index: d[1],
      }))
    },
    setDataTable(state, data) {
      const features = data.features
      const values = data.values
      state.data_header = features.map(d => ({ text: d, value: d }))
      const start = 10 ** Math.floor(Math.log(values[0].length - 1) / Math.log(10) + 1e-4)
      state.data_table = values[0].map((_, index) => {
        let _index = '' + index
        for (let thres = start; thres >= 10; thres /= 10) {
          if (index < thres) {
            _index = '0' + _index
          }
        }
        let ret = { id: `#${_index}`, _id: index }
        for (let j = 0; j < features.length; ++j) {
          ret[features[j]] = values[j][index]
        }
        return ret
      })
    },
    updateTooltip(state, attr) {
      for (let key in attr) {
        state.tooltipview[key] = attr[key]
      }
      if (state.tooltipview.content == '') {
        state.tooltipview.visibility = 'hidden'
      }
    },
    setNewScore(state, { new_scores }) {
      state.rules.forEach((d, i) => {
        d.anomaly = new_scores[i]
      })
    },
    changeRulefilter(state, filter) {
      state.rulefilter = filter
    },
    changeCrossfilter(state, filter) {
      state.crossfilter = filter
    },
    sortLayoutRow(state, key) {
      state.matrixview.sort_by_cover_num = !state.matrixview.sort_by_cover_num
    },
    setCurrentRule(state, x) {
      if (state.summary.current && state.summary.current.rule == x.rule) {
        state.summary.current.rule.is_selected = false
        state.summary.current = null
        state.summary.show_update = state.summary.update.changes != 0
      } else {
        if (state.summary.current) {
          state.summary.current.rule.is_selected = false
        }
        state.summary.current = x
        state.summary.current.rule.is_selected = true
        let code = state.summary.current.items.map(d => d.code).join(' & ').replaceAll(')(', ' & ')
        // console.log('current', x)
        // console.log('cond = ' + code)
        // state.summary.show_update = 0
        state.summary.show_update = state.summary.update.changes != 0
      }
    },
    setSuggestion(state, suggestion) {
      state.summary.suggestion = suggestion
    },
    focusLayoutCol(state, key) {
      if (!state.data_features[key]) return
      let exist = 0
      for (let i = 0; i < state.matrixview.focus_keys.length; ++i) {
        if (state.matrixview.focus_keys[i].key == key) {
          exist = 1
          state.matrixview.focus_keys.splice(i, 1)
        }
      }
      if (!exist) {
        state.matrixview.focus_keys.push({
          key: key, name: state.data_features[key].name
        })
      }
    },
    sortLayoutCol(state, key) {
      let exist = 0
      for (let i = 0; i < state.matrixview.order_keys.length; ++i) {
        if (state.matrixview.order_keys[i].key == key) {
          exist = 1
          if (state.matrixview.order_keys[i].order == 1) {
            state.matrixview.order_keys[i].order = -1
          } else if (state.matrixview.order_keys[i].order == 0) {
            state.matrixview.order_keys[i].order = 1
          } else {
            state.matrixview.order_keys.splice(i, 1)
          }
        }
      }
      if (!exist) {
        state.matrixview.order_keys.push({
          key: key,
          order: 1,
          name: state.data_features[key] ? state.data_features[key].name : key,
        })
      }
    },
    changePage(state, delta) {
      state.layout.fold_info.page += delta
      if (state.layout.fold_info.page >= state.layout.fold_info.n_pages) {
        state.layout.fold_info.page = state.layout.fold_info.n_pages - 1
      } else if (state.layout.fold_info.page < 0) {
        state.layout.fold_info.page = 0
      }
    },
    changePageSize(state, { width, height }) {
      state.page.width = width
      state.page.height = height
    },
    changeMatrixSize(state, { width, height }) {
      state.matrixview.width = width
      state.matrixview.height = height
    },
    ready(state, status) {
      state.is_ready = status
    },
    highlight_sample(state, sample_id) {
      if (state.highlighted_sample != sample_id) {
        state.highlighted_sample = sample_id
      } else {
        state.highlighted_sample = undefined
      }
    },
    updateMatrixLayout(state) {
      console.log('updateMatrixLayout')
      if (!state.is_ready) return

      const width = state.matrixview.width - state.matrixview.margin.left - state.matrixview.margin.right
      const height = state.matrixview.height - state.matrixview.margin.top - state.matrixview.margin.bottom
      const feature_base = [state.matrixview.feature.min_width, state.matrixview.feature.max_width]
      const max_level = Math.max(...state.rules.map(d => d.level))
      const min_level = Math.min(...state.rules.map(d => d.level))
      const zoom_level = max_level

      let features
      let rules = state.rules
      let fold_info
      if (zoom_level > state.matrixview.zoom_level) {
        fold_info = { page: 0, n_pages: 1, fold: 0 }
        state.stack.push(state.layout)
        features = JSON.parse(JSON.stringify(state.data_features))
        for (let i = 0; i < features.length; ++i) {
          const feature = features[i]
          feature.rules = rules.filter(d => d.conds.filter(cond => cond.key == feature.index).length)
          feature.count = feature.rules.length
          feature.hist = []
          for (let j = 0; j < state.dataset.label.length; ++j) {
            feature.hist.push(0)
          }
          feature.rules.forEach(d => feature.hist[d.predict] += 1)
        }
        features = features.sort((a, b) => b.importance - a.importance)
        fold_info.fold = 1
        if (zoom_level > 0) {
          const old_count = {}
          for (const feature of state.layout.features) {
            old_count[feature.name] = feature.count / state.layout.rules.length
          }

          features.forEach(d => {
            d.count_change = (d.count / rules.length) - old_count[d.name]
            d.pin = 0
          })
        }
        features.forEach(d => {
          if (d.count > 0) {
            d.show = 1
          } else {
            d.show = 1
          }
        })
        state.matrixview.zoom_level = zoom_level
      } else if (zoom_level < state.matrixview.zoom_level) {
        state.layout = state.stack.pop()
        state.matrixview.zoom_level = zoom_level
        return
      } else {
        features = state.layout.features
        fold_info = state.layout.fold_info
        features.forEach(d => {
          if (d.count > 0) {
            d.show = 1
          } else {
            d.show = 0
          }
        })
      }

      const hist_max = Math.max(...features.filter(d => d.show).map(d => Math.max(...d.hist)))
      const histScale = d3.scaleLinear()
        .domain([0, hist_max])
        .range([0, 25])

      const has_pin = features.filter(d => d.pin).length
      if (zoom_level >= state.matrixview.zoom_level) {
        const fold_items_per_page = state.matrixview.fold_items_per_page - has_pin + (zoom_level == 0 ? 8 : 0)
        const fold_features = features.filter(d => !d.pin)
        const changes = features
          //          .filter(d => d.count > 0)
          .map(d => d.count_change)
          .sort((a, b) => a - b)
        const increase_thres = Math.max(0.15, changes[changes.length - 3] || 0)
        const decrease_thres = Math.min(-0.15, changes[2] || 0)

        features.forEach(d => {
          if (d.count_change > increase_thres) {
            d.hint_change = 'arrow-up-thick'
          } else if (d.count_change < decrease_thres) {
            d.hint_change = 'arrow-down-thick'
          } else {
            d.hint_change = null
          }
        })

        if (zoom_level == 0) {
          if (fold_features.length < 20) {
            fold_info.n_pages = 1
          } else {
            fold_info.n_pages = 4
          }
          //TODO: the right pages
        } else {
          fold_info.n_pages = Math.ceil(fold_features.length / fold_items_per_page)
        }
        if (fold_info.n_pages == 1) {
          fold_info.fold = 0
          fold_info.next = 0
        } else {
          fold_info.next = fold_info.n_pages - fold_info.page - 1
          const start_index = fold_info.page * fold_items_per_page
          const end_index = start_index + fold_items_per_page
          fold_features.forEach((d, i) => { d.show = i >= start_index && i < end_index })
        }
        fold_info.has_left_page = fold_info.page > 0
        fold_info.has_right_page = fold_info.page + 1 < fold_info.n_pages
      }
      // console.log('foldinfo', fold_info)

      const rule_anomaly_delta = rules.map(d => d.anomaly - d.initial_anomaly).sort((a, b) => a - b)
      const hint_delta_min = Math.min(-0.1, rule_anomaly_delta[1] || 0)
      const hint_delta_max = Math.max(0.1, rule_anomaly_delta[rule_anomaly_delta.length - 2] || 0)
      rules.forEach(d => {
        if (d.labeled) {
          d.hint_change = 'star'
        } else if (d.anomaly - d.initial_anomaly <= hint_delta_min) {
          d.hint_change = 'arrow-down'
        } else if (d.anomaly - d.initial_anomaly >= hint_delta_max) {
          d.hint_change = 'arrow-up'
        } else {
          d.hint_change = ''
        }
      })

      rules.forEach(d => {
        if (d.is_selected) {
          d.show_hist = 1
        } else if (d.level == min_level) {
          d.represent = true
          d.show_hist = (zoom_level > 0)
        } else {
          d.represent = false
          d.show_hist = false
        }
      })
      let tot_hist = rules.filter(d => d.show_hist).length
      if (tot_hist >= 15) {
        rules.forEach(d => {
          d.show_hist = 0
        })
        tot_hist = 0
        rules.forEach(d => {
          if (d.is_selected) {
            d.show_hist = 1
            tot_hist += 1
          }
        })
      }

      rules = sortList(rules, zoom_level, state)

      const preserved_keys = new Set(state.matrixview.extended_cols.map(d => d.index));
      const orderkey_items = state.matrixview.order_keys.map(d => d.key)
        .filter(d => !preserved_keys.has(d))
      const orderkey_set = new Set(orderkey_items)

      let covered_samples = new Set()
      for (let rule of rules) {
        for (let id of rule.samples) {
          if (!covered_samples.has(id)) {
            covered_samples.add(id)
          }
        }
      }
      state.covered_samples = [...covered_samples]
      const max_count = Math.max(...features.map(d => d.count))
      const count_range = [max_count * state.matrixview.feature.hidden_percent, max_count]
      const samples = new Set([].concat(...rules.map(d => d.samples)))
      state.coverfilter = (d) => samples.has(d._id)
      state.primary.has_primary_key = 1

      const oldFeatureScale = d3.scaleLinear()
        .domain(count_range)
        .range(feature_base)

      let feature_sum = features.filter(d => d.show)
        .map(d => oldFeatureScale(d.count))
        .reduce((a, b) => a + b)

      const extended_cols = state.matrixview.extended_cols
      const n_extnded_features = features.filter(d => orderkey_set.has(d.index) || d.pin).length
      const focus_extend_width = n_extnded_features * state.matrixview.feature.extend_width
      const fold_left_gap = fold_info.fold && fold_info.page ? state.matrixview.fold_button_gap : state.matrixview.fold_normal_gap
      const fold_right_gap = fold_info.fold && fold_info.next ? state.matrixview.fold_button_gap : state.matrixview.fold_normal_gap
      const fold_gap = fold_left_gap + fold_right_gap
      const matrix_padding = state.matrixview.padding
      const coverage_width = state.matrixview.coverage_width

      const main_width = width
        - (matrix_padding + coverage_width) * extended_cols.length
        + 2 * state.matrixview.glyph_padding
        - focus_extend_width
        - fold_gap
      const width_ratio = main_width / feature_sum
      const main_start_x =
        (matrix_padding + coverage_width) * extended_cols.filter(d => d.position < 0).length
        + state.matrixview.glyph_padding + state.matrixview.glyph_width + 5
        + (has_pin ? 0 : fold_left_gap)
      const main_end_x = main_start_x
        + main_width + matrix_padding + focus_extend_width
        + fold_right_gap + (has_pin ? fold_left_gap : 0)
      const background_width = main_width + focus_extend_width


      const feature_range = [feature_base[0] * width_ratio, feature_base[1] * width_ratio]
      const featureScale = d3.scaleLinear()
        .domain(count_range)
        .range(feature_range)

      const coverage_range = d3.extent(rules, d => d.coverage)
      const max_coverage = Math.max(...rules.map(d => d.coverage))
      if (max_coverage > state.max_rule_coverage) {
        state.max_rule_coverage = max_coverage
      }
      // console.log('max_coverage', max_coverage, state.max_rule_coverage)
      const coverageScale = d3.scaleLinear()
        //        .domain([0, Math.max(...rules.map(d => d.coverage))])
        .domain([0, state.max_rule_coverage])
        .range([0, coverage_width * 0.75])

      let fidelityScale = d3.scaleLinear()
        .domain([0, 0.2, 0.8, 1])
        .range([0, coverage_width * 0.1, coverage_width * 0.9, coverage_width])
      //.range([coverage_width * 0.75, coverage_width])
      if (state.model_info.weighted) {
        // fidelityScale.range([coverage_width * 0.75, coverage_width])
      }

      const anomalyScale = d3.scaleLinear()
      //const anomalyScale = d3.scalePow()
        //.domain([Math.min(...rules.map(d => d.anomaly)), Math.max(...rules.map(d => d.anomaly))])
        .domain([0, 1])
        .range([0.075 * coverage_width, coverage_width])
        //.exponent(2.5)
      //.domain([0, 0.2, 0.9, 1])
      //.range([0, coverage_width*0.1, coverage_width*0.6, coverage_width])

      const weightScale = d3.scaleSqrt()
        .domain([Math.min(...rules.map(d => d.weight)), Math.max(...rules.map(d => d.weight))])
        .range([0.1 * coverage_width, coverage_width])

      const numScale = d3.scaleSqrt()
        .domain([0, Math.max(1, Math.min(...rules.map(d => d.num_children))), Math.max(...rules.map(d => d.num_children))])
        .range([.5, state.matrixview.min_indent_bar_size, state.matrixview.max_indent_bar_size])

      const rows = []
      let y = state.matrixview.row_padding
      if (rules.length * state.matrixview.row_height.large < height) {
        rules.forEach(d => d.show_hist = 1)
      }
      let lastheight = 0
      let hist_height = state.matrixview.row_height.large
      let instance_height = state.matrixview.row_height.medium
      const row_padding = state.matrixview.row_padding
      const tot_height = tot_hist * hist_height + (rules.length - tot_hist) * instance_height // state.matrixview.row_padding
      if (tot_height + row_padding * rules.length > height) {
        const ratio = height / tot_height
        hist_height *= ratio
        instance_height *= ratio
        hist_height -= row_padding
        instance_height -= row_padding
      }
      console.log('tot_height', tot_height)
      console.log('height', height)

      for (let i = 0; i < rules.length; ++i) {
        const rule = rules[i]
        const glyphheight = instance_height
        let height = instance_height//state.matrixview.row_height.medium
        if (rule.show_hist) {
          height = hist_height // state.matrixview.row_height.large
        }
        const x = 0//state.matrixview.glyph_padding//matrix_padding + coverage_width
        const _width = width - (matrix_padding + coverage_width) * 2
        const attrwidth = {
          num_children: rule.num_children,
          num: numScale(rule.num_children),
          coverage: coverageScale(rule.coverage),
          fidelity: fidelityScale(rule.fidelity),
          anomaly: anomalyScale(rule.anomaly),
          weight: weightScale(rule.weight),
        }
        const attrfill = {
          coverage: 'gray',
          fidelity: state.dataset.color.normal_color[rule.predict],
          anomaly: 'gray',
          weight: 'gray',
        } //'#7ec636' }
        rows.push({
          x,
          y,
          lastheight,
          height,
          glyphheight,
          rule,
          is_selected: rule.is_selected || false,
          fill: {
            bg: state.dataset.color.lighter_color[rule.predict],
            normal: state.dataset.color.normal_color[rule.predict],
            h: state.dataset.color.darker_color[rule.predict],
          },
          attrwidth, attrfill,
          hint_change: rule.hint_change,
          id: rule.id,
          samples: new Set(rule.samples)
        })
        lastheight = height
        y += height + row_padding
      }
      y += lastheight / 2

      const cols = []
      const indexed_cols = []
      const feature_padding = state.matrixview.cell.feature_padding

      for (let i = 0; i < features.length; ++i) {
        const feature = features[i]
        const has_key = orderkey_set.has(feature.index)
        const show_axis = has_key || feature.pin
        let width = featureScale(feature.count)
          + (show_axis ? state.matrixview.feature.extend_width : 0)
        if (!feature.show) {
          width = 0
        }
        if (show_axis &&
          feature.dtype == "category" &&
          feature.values.length * state.matrixview.feature.min_width_per_class > width) {
          const min_width = feature.values.length * state.matrixview.feature.min_width_per_class
          const delta = min_width - width
          //left_width -= delta

        }
        let range = feature.dtype == "category" ?
          [0, feature.values.length] :
          [Math.min(0, feature.range[0]), feature.range[1]]

        // TODO: handle distortion
        const distorted = false && feature.dtype != "category" && (
          feature.q[2] < (feature.range[1] - feature.range[0]) * 0.1 + feature.range[0])
        // || feature.q[2] > (feature.range[1] - feature.range[0]) * 0.9 + feature.range[0])
        const item = {
          width,
          height: height,
          index: feature.index,
          items: [],
          pin: feature.pin,
          name: feature.name,
          type: feature.dtype,
          count: feature.count,
          // feature histogram
          hist: feature.hist.map((d, j) => ({
            width: feature.hist.length <= 3 ? 9 : (30 / feature.hist.length),
            height: histScale(d),
            x: feature.hist.length == 2 ? 
            (9 + (width - 26) / (feature.hist.length - 1) * j) :
            (6 + ((width - 20) / (feature.hist.length - 1) + 1) * j),
            y: 33 - histScale(d) + height,
            fill: state.dataset.color.color_scheme[j],
          })),
          display_name: feature.display_name,
          hint_change: feature.hint_change,
          delta: feature.hint_change ? 1 : 0,
          range,
          q: feature.q,
          distorted,
          scale: feature.scale,
          values: feature.values,
          is_glyph: 0,
          show_axis,
          show: feature.show,
        }
        cols.push(item)
        indexed_cols[item.index] = item
      }

      const tot_width = cols.filter(d => !d.show_axis).map(d => d.width).reduce((a, b) => a + b)
      let left_width = tot_width

      for (let i = 0; i < cols.length; ++i) {
        if (cols[i].show_axis && cols[i].type == "category") {
          const min_width = cols[i].values.length * state.matrixview.feature.min_width_per_class
          if (min_width > cols[i].width) {
            const delta = min_width - cols[i].width
            left_width -= delta
            cols[i].width = min_width
          }
        }
      }
      fold_info.left_x = main_start_x
      fold_info.right_x = background_width + main_start_x

      for (let i = 0, x = main_start_x; i < cols.length; ++i) {
        const col = cols[i]
        col.x = x
        col.y = 0
        if (!col.show_axis && left_width < tot_width) {
          col.width *= left_width / tot_width
        }
        col.display_range = [0, col.width - feature_padding * 2]
        if (col.distorted && col.range[0] >= 0) {
          col.scale = makeScale('log', col.range, col.display_range)
        } else if (col.scale == 'log') {
          col.scale = makeScale('sqrt', col.range, col.display_range)
        } else {
          col.scale = makeScale('linear', col.range, col.display_range)
        }
        x += cols[i].width
        if (cols[i].pin && i < cols.length - 1 && !cols[i + 1].pin) {
          fold_info.pin_width = x - main_start_x
          x += fold_left_gap
          fold_info.left_x = x
          fold_info.right_x += focus_extend_width + fold_left_gap
        }
      }

      /*
      if (has_pin) {
        x += current_fold_width
      }*/

      for (let i = 0; i < extended_cols.length; ++i) {
        const width = coverage_width
        let x = 0
        if (extended_cols[i].position < 0) {
          x = main_start_x + extended_cols[i].position * (coverage_width + matrix_padding)
        } else {
          x = main_end_x + extended_cols[i].position * (coverage_width + matrix_padding)
        }
        const item = {
          x, y: 0, width, height, index: extended_cols[i].index, name: extended_cols[i].name, show: 1,
        }
        cols.push(item)
        indexed_cols[item.index] = item
      }

      let all_text = ''
      rows.forEach((row, row_index) => {
        // row.height = instance_height
        row.width = state.matrixview.width - state.matrixview.margin.left - 8
        row.items = row.rule.conds.filter(d => indexed_cols[d.key])
          .map((d, i) => {
            const feature = indexed_cols[d.key]
            let elements = []
            if (feature.type == 'category') {
              const s = d.range.reduce((a, b) => a + b)
              const neg = 0//s > d.range.length / 2
              const cond1 = row.rule.represent && zoom_level > 0
              const left_delta = (row.rule.show_hist ? -feature_padding : feature_padding) * 0.75
              const right_delta = (row.rule.show_hist ? feature_padding : -feature_padding) * 0.75
              for (let j = 0; j < d.range.length; ++j) {
                const cond2 = (d.range[j] > 0) != neg
                if (cond1 && (d.range[j] > 0) || cond2 && !cond1)
                  elements.push({
                    x0: feature.scale(j) + left_delta,
                    x1: feature.scale(j + 1) + right_delta,
                    show: feature.show,
                    h: row.height,
                    show_hist: row.rule.show_hist,
                    fill: row.fill.normal,
                    neg: !cond1 && neg
                  })
                else {
                  elements.push({
                    x0: feature.scale(j) + left_delta,
                    x1: feature.scale(j + 1) + right_delta,
                    show: feature.show,
                    h: row.height,
                    show_hist: row.rule.show_hist,
                    fill: row.fill.bg,
                  })
                }
              }
            } else {
              const min_gap = state.matrixview.bar_min_width
              let x0 = feature.scale(d.range[0])
              let x1 = feature.scale(d.range[1])
              if (x0 <= feature.display_range[0] + min_gap) {
                x0 = feature.display_range[0]
              }
              if (x1 >= feature.display_range[1] - min_gap) {
                x1 = feature.display_range[1]
              }

              if (x0 + min_gap > x1) {
                let delta = x0 + min_gap - x1
                if (x0 + min_gap >= feature.width) {
                  delta = feature.width - x1
                }
                if (x1 < min_gap) {
                  delta = x0
                }
                x0 -= delta / 2
                x1 += delta / 2
                if (x1 - x0 < min_gap) {
                  if (x0 == 0) {
                    x1 += (min_gap - (x1 - x0)) / 2
                  } else {
                    x0 -= (min_gap - (x1 - x0)) / 2
                  }
                }
              } else if (x1 - x0 + min_gap > feature.width) {
                if (d.range[1] >= feature.range[1]) {
                  x0 = x1 - feature.width + min_gap
                } else {
                  x1 = x0 + feature.width - min_gap
                }
              }
              x0 += feature_padding
              x1 += feature_padding
              //todo
              elements.push({
                x0: x0,
                x1: x1,
                h: row.height,
                show: feature.show,
                show_hist: row.rule.show_hist,
                fill: row.fill.normal,
                hfill: row.fill.h,
              })
            }

            return {
              scale: feature.scale,
              elements,
              x: feature.x,
              y: row.y,
              show: feature.show,
              is_glyph: 0,
              width: feature.width,
              height: row.height,
              fill: row.fill.normal,
              bgfill: row.fill.bg,
              hfill: row.fill.h,
              cond: row.rule.conds[i],
              name: feature.name,
              id: row.id,
              feature: feature,
              represent: row.rule.represent,
              show_hist: row.rule.show_hist,
              samples: row.rule.show_hist ? [...row.samples] : []
            }
          })
        // console.log('row items', row.items)
        row.items.forEach(d => {
          let code = ''
          if (d.feature.type == 'category') {
            const s = d.cond.range.reduce((a, b) => a + b)
            let text = `${d.name} is `
            let items = []
            if (s <= d.cond.range.length / 2) {
              for (let i = 0; i < d.cond.range.length; ++i) {
                if (d.cond.range[i]) {
                  items.push(d.feature.values[i])
                  if (typeof d.feature.values[i] === 'number')
                    code += `(data['${d.name}'] == ${d.feature.values[i]})`
                  else
                    code += `(data['${d.name}'] == '${d.feature.values[i]}')`
                }
              }
            } else {
              text += 'NOT '
              for (let i = 0; i < d.cond.range.length; ++i) {
                if (!d.cond.range[i]) {
                  items.push(d.feature.values[i])
                  if (typeof d.feature.values[i] === 'number')
                    code += `(data['${d.name}'] != ${d.feature.values[i]})`
                  else
                    code += `(data['${d.name}'] != '${d.feature.values[i]}')`
                }
              }
            }
            text += items.join(', ')
            d.text = text

          } else {
            const precision = parseInt(state.dataset.format)
            const exp10 = 10 ** precision
            const left = Math.round(Math.max(d.feature.range[0], d.cond.range[0]) * exp10) / exp10
            const right = Math.round(Math.min(d.feature.range[1], d.cond.range[1]) * exp10) / exp10
            d.text = ''
            if (d.cond.range[0] > d.feature.range[0] && d.cond.range[1] < d.feature.range[1]) {
              d.text += `${left} < `
              code += `(data['${d.name}'] > ${left})`
              d.text += ` ${d.name} `
              d.text += ` <= ${right}`
              code += `(data['${d.name}'] <= ${right})`
            } else if (d.cond.range[0] > d.feature.range[0]) {
              d.text += ` ${d.name} `
              d.text += ` > ${left} `
              code += `(data['${d.name}'] > ${left})`
            } else {
              d.text += ` ${d.name} `
              d.text += ` <= ${right}`
              code += `(data['${d.name}'] <= ${right})`
            }
            if (row.rule.missing && row.rule.missing.indexOf(+d.cond.key) != -1) {
              d.text += ' or missing'
            }
          }
          d.code = code
        })
        if (row.rule.show_hist) {
          const exist_keys = new Set(row.items.map(d => d.name))
          row.items = row.items.concat(features
            .filter(d => !exist_keys.has(indexed_cols[d.index].name))
            .map(d => ({
              scale: indexed_cols[d.index].scale,
              elements: [],
              x: indexed_cols[d.index].x,
              y: row.y,
              show: indexed_cols[d.index].show,
              is_glyph: 0,
              width: indexed_cols[d.index].width,
              height: row.height,
              fill: row.fill.normal,
              hfill: row.fill.h,
              cond: [],
              name: indexed_cols[d.index].name,
              id: row.id,
              feature: indexed_cols[d.index],
              represent: row.rule.represent,
              show_hist: row.rule.show_hist,
              samples: [...row.samples]
            }))
          )
          // console.log('row.items', row.items)
        }
        row.attr = { num: row.attrwidth.num, num_children: row.attrwidth.num_children }
        row.extends = extended_cols.map((d, i) => ({
          name: d.index,
          x1: indexed_cols[d.index].x,
          x2: indexed_cols[d.index].x + row.attrwidth[d.index],
          x: indexed_cols[d.index].x,
          y: row.y,
          show: indexed_cols[d.index].show,
          width: indexed_cols[d.index].width,
          height: Math.min(instance_height, row.height),
          fill: row.attrfill[d.index],
          value: row.rule[d.index],
          represent: row.rule.represent,
          text: Number(row.rule[d.index]).toFixed(3)
        }))
        row.extends.forEach(d => {
          if (d.name == 'anomaly' && Math.abs(row.rule.anomaly - row.rule.initial_anomaly) > 0.1) {
            const delta = row.rule.anomaly - row.rule.initial_anomaly
            d.text += `(${delta > 0 ? '+' : '-'}${Number(Math.abs(delta)).toFixed(3)})`
          } else if (d.name == 'coverage') {
            d.text = '' + Number(row.rule.coverage).toFixed(3)
          }
        })
        let current_text = '#' + row_index + ' if'
        for (let i = 0; i < row.items.length; ++i) {
          let d = row.items[i]
          current_text += ' ' + d.text + ((i == row.items.length - 1) ? ' then' : ' and')
        }
        current_text += ' ' + state.dataset.label[row.rule.predict]
        all_text += current_text + '\n\n'
      })
      // console.log('cols', cols)

      state.layout = {
        fold_info,
        has_pin,
        main_start_x,
        main_width: background_width,
        features,
        rules,
        cols,
        rows,
        height: y,
        width,
      }
    },
    changeFeaturePosition(state, { name, target }) {
      console.log('move', name, target)
      let new_importance = 0
      state.layout.features.forEach((d, i) => {
        if (d.name == target) {
          if (i == state.layout.features.length - 1) {
            new_importance = d.importance * 0.5
          } else {
            new_importance = (d.importance + state.layout.features[i + 1].importance) / 2
          }
        }
      })
      state.layout.features.forEach((d, i) => {
        if (d.name == name) {
          d.importance = new_importance
        }
      })
      state.data_features.forEach((d, i) => {
        if (d.name == name) {
          d.importance = new_importance
        }
      })
      const pined = state.layout.features.filter(d => d.pin).sort((a, b) => b.importance - a.importance)
      pined.forEach(d => d.show = 1)
      const unpined = state.layout.features.filter(d => !d.pin).sort((a, b) => b.importance - a.importance)
      state.layout.features = pined.concat(unpined)
    },
    changeFeaturePin(state, name) {
      state.layout.features.forEach(d => {
        if (d.name == name) {
          d.pin = !d.pin
          if (d.pin) {
            d.old_importance = d.importance
            d.importance = 1
          } else {
            d.importance = d.old_importance
          }
        }
      })
      const pined = state.layout.features.filter(d => d.pin).sort((a, b) => b.importance - a.importance)
      pined.forEach(d => d.show = 1)
      const unpined = state.layout.features.filter(d => !d.pin).sort((a, b) => b.importance - a.importance)
      state.layout.features = pined.concat(unpined)
    },
    /*
    setAllSamples(state, data) {
      state.samples = data.sort((a, b) => a.id - b.id)
    },
    setAllRules(state, data) {
      state.rules = data.sort((a, b) => a.id - b.id)
    },
    */
    // 左端点 + 右端点同时考虑在内
    // primary + secondary 双排序
    // filter by one feature (age)
    // sample background - switch - typical sample
    // more filter / legend
    // zoom in with more space, line to encoding a sample
    // star to encoding represent rules
    // anomaly => Anomaly Score
    // categorical data - explainable matrix没有，批评
    // one-hot: important. vs others.
    // representative rules need to be highlighted, 柠檬黄, stroke, extend in the front
    // interaction design
    setRulePaths(state, { paths, samples, info }) {
      state.summary.info = state.summary.update = info
      Object.assign(state.summary, { suggestion: null })
      const raw_rules = paths.map((rule) => ({
        distribution: rule.distribution,
        labeled: rule.labeled,
        id: rule.name,
        tree_index: rule.tree_index,
        rule_index: rule.rule_index,
        feature: rule.feature_vector,
        coverage: rule.coverage,
        anomaly: rule.anomaly,
        initial_anomaly: rule.initial_anomaly,
        real_idx: rule.idx,
        weight: rule.weight,
        level: rule.level,
        loyalty: (rule.coverage ** 0.5) * rule.anomaly,
        father: rule.father,
        fidelity: rule.confidence,
        cond_dict: rule.range,
        num_children: rule.num_children || 0,
        predict: rule.output,
        range: rule.range,
        q: rule.q,
        samples: rule.samples,
        missing: rule.missing,
        cond_norm: rule.cond_norm,
        conds: Object.keys(rule.range).map(cond_key => ({
          key: cond_key,
          range: rule.range[cond_key],
        })).filter(d => state.data_features[d.key].dtype != 'number'
          || d.range[0] > -1e17//state.data_features[d.key].range[0]
          || d.range[1] < 1e17)//state.data_features[d.key].range[1])
      }))//.filter(d => d.fidelity > 0.6)
      console.log('total rules', raw_rules.map(d => d.num_children).reduce((a, b) => a + b))

      raw_rules.forEach((rule) => {
        rule.cond_sort = {}
        for (let key in rule.cond_norm) {
          if (state.data_features[key].dtype == 'category') {
            rule.cond_sort[key] = rule.cond_norm[key]
          } else {
            rule.cond_sort[key] = [
              state.data_features[key]._scale(rule.range[key][0]), 
              state.data_features[key]._scale(rule.range[key][1])
            ]
          }
        }
      })
      
      if (state.matrixview.max_level == -1) {
        state.matrixview.max_level = Math.max(...raw_rules.map(d => d.level))
      }

      for (let feature of state.data_features) {
        feature.importance = 0
      }

      for (let rule of raw_rules) {
        // rule.level = state.matrixview.max_level - rule.level
        rule.range_key = {}
        for (let key of Object.keys(rule.range)) {
          state.data_features[key].importance += 1
          const range = rule.range[key]
          if (range.length == 2) {
            rule.range_key[key] = (range[0] * 1e8 + range[1])
          } else {
            const neg = 0 // range.reduce((a, b) => a + b) > range.length / 2
            if (neg) {
              rule.range_key[key] = range.map(d => d ? '0' : '2').join('')
            } else {
              rule.range_key[key] = range.map(d => d ? '1' : '0').join('')
            }
          }
        }
      }

      const max_importance = Math.max(...state.data_features.map(d => d.importance))
      for (let feature of state.data_features) {
        feature.importance /= max_importance
        feature.importance = feature.importance ** 3
      }
      console.log(state.data_features.map(d => ({ name: d.name, importance: d.importance})))

      raw_rules.forEach(d => d.selected = 1)
      state.rules = raw_rules
      state.summary.number_of_rules = raw_rules.length
    },
    setZoomStatus(state, status) {
      state.matrixview.zoom_level = status
    },
    setFeatures(state, features) {
      const raw_features = features.map(
        (feature, feature_index) => ({
          index: feature_index,
          id: `F${feature_index}`,
          importance_: feature.importance,
          range: feature.range,
          q: feature.q,
          name: feature.name,
          dtype: feature.dtype,
          values: feature.values,
          default: feature.default,
          display_name: feature.display_name || null,
          scale: feature.scale,
          _scale: feature.dtype == 'category' ? null : makeScale(feature.scale, feature.range, [0, 1]),
        }))
      raw_features.forEach((d, index) => {
        d.selected = 1
      })
      state.data_features = raw_features
    },
    setInstances(state, data) {
      state.instances = data
    },
    changeRuleLabel(state, name) {
      state.rules.forEach(d => {
        if (d.id == name) {
          d.labeled = 1
        }
      })
    },
    summaryDataInfo(state, data) {
      const dataset = state.dataset
      const positives = data.filter(d => d['Label'] == dataset.label[1]).length;
      const corrects = data.filter(d => d['Label'] == d.Predict).length;
      const total = data.length
      const prob = positives / total
      state.summary.update = {
        positives,
        prob,
        total,
        change_sgn: prob - state.summary.info.prob > 0 ? '+' : '-',
        changes: Math.abs(prob - state.summary.info.prob),
        accuracy: corrects / data.length,
      }
      state.summary.show_update = state.summary.update.changes != 0
    }
  },
  getters: {
    model_target: (state) => 'Label',//state.model_info.target,
    zoom_level: (state) => state.matrixview.zoom_level,
    //model_info: 'Dataset.name: German credit, Model: Random Forest, Original Accuracy: 82.83%, Fedility: 95.20%',
    model_info: (state) => `Dataset: ${state.model_info.dataset}, Model: ${state.model_info.model}, Original Accuracy: ${Number(state.model_info.accuracy * 100).toFixed(2)}%, Fedility: ${Number(state.model_info.info.fidelity_test * 100).toFixed(2)}%`,
    rule_info: (state) => `${state.layout.rows.length} out of ${state.model_info.num_of_rules} rules`,
    view_info: (state) => `${state.matrixview.zoom_level > 0 ? ('Zoom level ' + state.matrixview.zoom_level) : 'Overview'}`,
    topview_height: (state) => state.matrixview.height,
    filtered_data: (state) => {
      if (state.matrixview.zoom_level > 0) {
        const covered_samples = new Set(state.covered_samples)
        //const ret = state.data_table.filter(d => state.crossfilter(d))
        const ret = state.data_table.filter(d => state.crossfilter(d) && covered_samples.has(d._id))
        return ret
      } else {
        // console.log('state.data_table', state.data_table)
        const ret = state.data_table.filter(d => state.crossfilter(d))
        return ret
      }
    },
    rule_related_data: (state) => {
      const covered_samples = new Set(state.covered_samples)
      return state.data_table.filter(d => covered_samples.has(d._id))
    },
  },
  actions: {
    async showRules({ commit, state }, data) {
      let resp
      if (data == null && state.matrixview.last_show_rules.length > 1) {
        data = state.matrixview.last_show_rules.splice(-2, 1)[0]
      } else if (data != null) {
        state.matrixview.last_show_rules.push(data)
      }
      if (data == null) {
        state.matrixview.last_show_rules = []
        resp = await axios.post(`${state.server_url}/api/selected_rules`, {
          dataname: state.dataset.name, session_id: state.session_id
        })
      } else {
        resp = await axios.post(`${state.server_url}/api/explore_rules`, { dataname: state.dataset.name, session_id: state.session_id, idxs: data, N: ~~(state.matrixview.n_lines) })
      }
      commit('setRulePaths', resp.data)
      commit('updateMatrixLayout')
    },
    async orderRow({ commit }, data) {
      commit('sortLayoutRow', data)
      commit('updateMatrixLayout')
    },
    async updatePage({ commit }, delta) {
      commit('changePage', delta)
      commit('updateMatrixLayout')
    },
    async orderColumn({ commit }, data) {
      commit('sortLayoutCol', data)
      commit('updateMatrixLayout')
    },
    async fetchRawdata({ commit, state }) {
      let resp = await axios.post(`${state.server_url}/api/model_info`, { dataname: state.dataset.name })
      commit('setModelInfo', resp.data)
      resp = await axios.post(`${state.server_url}/api/features`, { dataname: state.dataset.name, session_id: state.session_id })
      commit('setFeatures', resp.data)
      resp = await axios.post(`${state.server_url}/api/selected_rules`, { dataname: state.dataset.name, session_id: state.session_id })
      commit('setRulePaths', resp.data)
      resp = await axios.post(`${state.server_url}/api/data_table`, { dataname: state.dataset.name, session_id: state.session_id, precision: state.dataset.format })
      commit('setDataTable', resp.data)
      commit('setInstances', [])
    },
    async updateMatrixLayout({ commit }) {
      commit('updateMatrixLayout')
    },
    async setReady({ commit }) {
      commit('ready', true)
    },
    async setUnready({ commit }) {
      commit('ready', true)
    },
    async updateMatrixWidth({ commit }, width) {
      commit('changeMatrixWidth', width)
    },
    async updateMatrixSize({ commit }, { width, height }) {
      commit('changeMatrixSize', { width, height })
      commit('updateMatrixLayout')
    },
    async updatePageSize({ commit }, { width, height }) {
      commit('changePageSize', { width, height })
    },
    async updateRulefilter({ commit, state }, filter) {
      commit('changeRulefilter', filter)
      // commit('updateMatrixLayout')
    },

    async updateCurrentRule({ commit }, rule) {
      commit('setCurrentRule', rule)
      commit('updateMatrixLayout')
    },
    async updateCrossfilter({ commit, state }, filter) {
      commit('changeCrossfilter', filter)
      // commit('updateMatrixLayout')
    },
    async updateDataInfo({ commit }, data) {
      commit('summaryDataInfo', data)
    },
    async updateFeaturePin({ commit }, name) {
      commit('changeFeaturePin', name)
      commit('updateMatrixLayout')
    },
    async updateFeaturePosition({ commit },{ name, target }) {
      commit('changeFeaturePosition', { name, target })
      commit('updateMatrixLayout')
    },
    highlightSample({ commit }, sample_id) {
      commit('highlight_sample', sample_id)
    },
    async tooltip({ commit }, { type, data }) {
      if (type == 'show') {
        commit('updateTooltip', { visibility: 'visible' })
      } else if (type == 'hide') {
        commit('updateTooltip', { visibility: 'hidden' })
      } else if (type == 'text') {
        commit('updateTooltip', { content: data })
      } else if (type == 'position') {
        commit('updateTooltip', { x: data.x + 10, y: data.y - 50 })
      }
    },
    async updateRuleLabel({ state, commit }, { name, label }) {
      let resp = await axios.post(`${state.server_url}/api/adjust_label`, {
        dataname: state.dataset.name, session_id: state.session_id,
        name: name,
        label: label,
        selected_indexes: state.rules.map(d => d.id),
      })
      commit('changeRuleLabel', name)
      commit('setNewScore', resp.data)
      commit('updateMatrixLayout')
    },
    async findSuggestion({ state, commit }, target) {
      let resp = await axios.post(`${state.server_url}/api/suggestions`, {
        dataname: state.dataset.name, session_id: state.session_id,
        ids: state.rules.map(d => d.id),
        target: target,
      })
      commit('setSuggestion', resp.data)
    },
    async clearSession({ state }) {
      let resp = await axios.post(`${state.server_url}/api/clear_session`, {
        session_id: state.session_id
      })
      location.reload();
    },
    async getDistribution({ state }, { id, feature }) {
      const key = `${id}_${feature}`
      if (distribution_cache[key]) {
        return distribution_cache[key]
      } else {
        const resp = await axios.post(`${state.server_url}/api/distribution`, {
          dataname: state.dataset.name,
          session_id: state.session_id,
          id, feature,
        })
        distribution_cache[key] = resp.data
        return resp.data
      }
    }
  },
  modules: {
  }
})
