import * as d3 from 'd3'
import { mdiArrowUpThick, mdiArrowDownThick, mdiArrowUpBold, mdiArrowDownBold, mdiStar } from '@mdi/js'
import { mdiReply, mdiPin, mdiMenuLeft, mdiMenuRight, mdiDotsHorizontal, mdiCircleSmall } from '@mdi/js'

const default_color = {
  lighter_color: [
    d3.interpolateLab('white', '#f28e2c')(0.15),
    d3.interpolateLab('white', '#4e79a7')(0.15),
    d3.interpolateLab('white', '#e15759')(0.15),
    d3.interpolateLab('white', '#76b7b2')(0.15),
    d3.interpolateLab('white', '#59a14f')(0.15),
    d3.interpolateLab('white', '#edc949')(0.15),
    d3.interpolateLab('white', '#af7aa1')(0.15),
    d3.interpolateLab('white', '#ff9da7')(0.15),
  ],
  normal_color: [
    d3.interpolateLab('white', '#f28e2c')(1),
    d3.interpolateLab('white', '#4e79a7')(1),
    d3.interpolateLab('white', '#e15759')(1),
    d3.interpolateLab('white', '#76b7b2')(1),
    d3.interpolateLab('white', '#59a14f')(1),
    d3.interpolateLab('white', '#edc949')(1),
    d3.interpolateLab('white', '#af7aa1')(1),
    d3.interpolateLab('white', '#ff9da7')(1),
  ],
  darker_color: [
    'rgb(246,227,100)',
    'rgb(183,201,237)',
    d3.interpolateLab('white', '#e15759')(0.6),
    d3.interpolateLab('white', '#76b7b2')(0.6),
    d3.interpolateLab('white', '#59a14f')(0.6),
    d3.interpolateLab('white', '#edc949')(0.6),
    d3.interpolateLab('white', '#af7aa1')(0.6),
    d3.interpolateLab('white', '#ff9da7')(0.6),
  ],
  color_scheme: [
    d3.interpolateLab('white', '#f28e2c')(1),
    d3.interpolateLab('white', '#4e79a7')(1),
    d3.interpolateLab('white', '#e15759')(1),
    d3.interpolateLab('white', '#76b7b2')(1),
    d3.interpolateLab('white', '#59a14f')(1),
    d3.interpolateLab('white', '#edc949')(1),
    d3.interpolateLab('white', '#af7aa1')(1),
    d3.interpolateLab('white', '#ff9da7')(1),
  ],
}

//const stock_color_scheme = ['#8bc0a6', '#e1d463', '#db8073']
//const stock_color_scheme = ['#8bc0a6', '#e1d463', 'rgb(224,27,114)']
//const stock_color_scheme = ['rgb(43,142,64)', '#e1d463', 'rgb(224,27,114)']
//const stock_color_scheme = ['rgb(43,142,64)', 'rgb(243,171,22)', 'rgb(224,27,114)']
//const stock_color_scheme = ['rgb(43,142,64)', 'rgb(243,171,22)', 'rgb(231,43,50)']
//const stock_color_scheme = ['rgb(36,140,95)', 'rgb(159,129,87)', 'rgb(231,43,50)']
// const stock_color_scheme = ['rgb(36,140,95)', 'rgb(243,171,22)', 'rgb(231,43,50)']


const stock_color_scheme_real = ['rgb(55,160,47)', 'rgb(255,217,102)', 'rgb(192,0,0)']
const stock_color_scheme = ['rgb(192,0,0)', 'rgb(55,160,47)', 'rgb(255,217,102)', ]
//['#828F56', '#397FE6', '#C75242']

const stock_color = {
  lighter_color: stock_color_scheme.map(d => d3.interpolateLab('white', d)(0.1)),
  normal_color: stock_color_scheme.map(d => d3.interpolateLab('white', d)(1)),
  darker_color: stock_color_scheme.map(d => d3.interpolateLab('white', d)(0.5)),
  color_scheme: stock_color_scheme.map(d => d3.interpolateLab('white', d)(1)),
}


const datasets = [{
  name: 'credit0',
  format: '1',
  target: 'Approved',
  label: ['reject', 'accept'],
  label0: ['No', 'Yes'],
  color: default_color,
}, {
  name: 'credit1',
  format: '1',
  target: 'Approved',
  label: ['reject', 'accept'],
  label0: ['No', 'Yes'],
  color: default_color,
}, {
  name: 'credit2',
  format: '1',
  target: 'Approved',
  label: ['reject', 'accept'],
  label0: ['No', 'Yes'],
  color: default_color,
}, {
  name: 'stock',
  format: '4',
  target: 'analystConsensus',
  label: ["Increase", "Stable", "Decrease"],
  label0: ["Increase", "Stable", "Decrease"],
  color: stock_color,
}, {
  name: 'stock1',
  format: '4',
  target: 'analystConsensus',
  label: ["Increase", "Stable", "Decrease"],
  label0: ["Increase", "Stable", "Decrease"],
  color: stock_color,
}, {
  name: 'stock2',
  format: '4',
  target: 'analystConsensus',
  label: ["Increase", "Stable", "Decrease"],
  label0: ["Increase", "Stable", "Decrease"],
  color: stock_color,
}, {
  name: 'stock3',
  format: '4',
  target: 'analystConsensus',
  label: ["Increase", "Stable", "Decrease"],
  label0: ["Increase", "Stable", "Decrease"],
  color: stock_color,
}, {
  name: 'german',
  format: '0',
  target: 'credit_risk',
  label: ['reject', 'accept'],
  label0: ['No', 'Yes'],
  color: default_color,
}, {
  name: 'drybean',
  format: '2',
  target: 'Class',
  label: ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'],
  label0: ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'],
  color: default_color,
}, {
  name: 'drybean1',
  format: '2',
  target: 'Class',
  label: ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'],
  label0: ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'],
  color: default_color,
}, {
  name: 'obesity',
  format: '2',
  target: 'NObeyesdad',
  label: ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"],
  label0: ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"],
  color: default_color,
}, {
  name: 'obesity1',
  format: '2',
  target: 'NObeyesdad',
  label: ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"],
  label0: ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"],
  color: default_color,
}]

const state = {
    server_url: 'http://166.111.81.51:5001',
    layout: null,
    primary: {
      key: null,
      order: -1,
    },
    session_id: '',
    debug: true,
    is_ready: false,
    rulefilter: () => 1,
    crossfilter: () => 1,
    coverfilter: () => 1,
    highlighted_sample: undefined,
    instances: [],
    dataset: datasets[0],
    stack: [],
    dataset_candidates: datasets,
    data_features: [],
    data_table: [],
    data_header: [],
    model_info: { loading: false },
    glyph: {
      'marker-more': {
        path: 'M7.41,8.58L12,13.17L16.59,8.58L18,10L12,16L6,10L7.41,8.58Z',
        fill: '#333',
      },
      'marker-expand': {
        path: 'M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z',
        fill: '#333',
      },
      'arrow-up': {
        path: mdiArrowUpBold,
        fill: '#555',
      },
      'arrow-up': {
        path: mdiArrowUpBold,
        fill: '#555',
      },
      'arrow-down': {
        path: mdiArrowDownBold,
        fill: '#555',
      },
      'star': {
        path: mdiStar,
        fill: '#555',
      },
      'arrow-up-thick': {
        path: mdiArrowUpThick,
        fill: '#555',
      },
      'arrow-down-thick': {
        path: mdiArrowDownThick,
        fill: '#555',
      },
      'back': {
        path: mdiReply,//'M280-200v-80h284q63 0 109.5-40T720-420q0-60-46.5-100T564-560H312l104 104-56 56-200-200 200-200 56 56-104 104h252q97 0 166.5 63T800-420q0 94-69.5 157T564-200H280Z', //mdiUndo,
        fill: '#555',
      },
      'pin': {
        path: mdiPin,
        fill: '#555',
      },
      'prev-page': {
        path: mdiMenuLeft,
        fill: '#555',
      },
      'next-page': {
        path: mdiMenuRight,
        fill: '#555',
      },
      '1dot': {
        path: mdiCircleSmall,
        fill: '#555',
      },
      '3dots': {
        path: mdiDotsHorizontal,
        fill: '#555',
      },
    },
    page: { width: 1800, height: 1000 },
    covered_samples: [],
    max_rule_coverage: 0.01,
    summary: {
      current: null,
      info: null,
      show_update: false,
      update: { changes: 0 },
      suggestion: null,
    },
    matrixview: {
      maxlen: 18,
      font_size: 16,
      indent_size: 30,
      max_indent_bar_size: 60,
      min_indent_bar_size: 10,
      fold_items_per_page: 10,
      fold_normal_gap: 18,
      fold_button_gap: 36,
      max_level: -1,
      sort_by_cover_num: true,
      n_lines: 80,
      padding: 10,
      cell_padding: 3,
      max_columns: 40,
      last_show_rules: [],
      margin: {
        top: 130,
        right: 180,
        bottom: 60,
        left: 75,
      },
      width: 1500,
      height: 800,
      coverage_width: 50,
      glyph_width: 60,
      glyph_padding: 35,
      bar_min_width: 5,
      duration: 800,
      cell: {
        feature_padding: .5,
        header_opacity: .5,
        highlight_header_opacity: .8,
        stroke_width: 1,
        highlight_stroke_width: 1,
        stroke_color: 'darkgray',
        highlight_stroke_color: 'black'
      },
      feature: {
        extend_width: 60,
        min_width_per_class: 23,
        max_width: 60,
        min_width: 60,
        hidden_percent: 0.15,
        glyph_width: 25,
      },
      rule_size_max: 1,
      hist_height: 2.5,
      order_keys: [],
      focus_keys: [],
      zoom_level: -1,
      extended_cols: [],
      row_height: {
        small: 2,
        medium: 6.3,
        large: 28.5,
      },
      row_padding: 2,
    },
    featureview: {
      textwidth: 125,
      maxlen: 16,
      fontweight: 400,
      padding: 15,
      column_height: 55,
      chart_height: 20,
      scrollbar_width: 10,
      handle_color: '#666666',
      glyph_color: '#d62728',
      bar_color: '#888',
      highlight_color: d3.color('#df5152').darker(-0.3),
    },
    tableview: {
      height: 250
    },
    tooltipview: {
      width: 300,
      content: '',
      visibility: 'hidden',
      x: 100,
      y: 100,
    },
}

export default state;
