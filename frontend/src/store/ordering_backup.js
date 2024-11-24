class RandomGenerator {
  constructor(seed) {
    this.seed = seed;
    this.modulus = 4294967291;
    this.multiplier = 1664525;
    this.increment = 1013904223;
  }

  next() {
    this.seed = (this.multiplier * this.seed + this.increment) % this.modulus;
    return this.seed;
  }

  random() {
    return this.next() / this.modulus;
  }
}

function calculateFeatureDistance(point1, point2, i, default_value) {
  let distance = 0
  const a = point1.cond_norm[i] || default_value
  const b = point2.cond_norm[i] || default_value
  let d = 0
  if (point1.predict == point2.predict) {
    for (let j = 0; j < a.length; ++j) {
      d += (a[j] - b[j]) * (a[j] - b[j])
    }
  } else {
    d = 1
  }
  distance += Math.sqrt(d)
  return distance
}

function calculateDistance(point1, point2, features, min_importance) {
  let distance = 0;

  const keys = [...new Set(Object.keys(point1.cond_norm).concat(Object.keys(point2.cond_norm)))]
  for (let i of keys) if (features[i].importance >= min_importance) {
    const a = point1.cond_norm[i] || features[i].default
    const b = point2.cond_norm[i] || features[i].default
    let d = 0
    for (let j = 0; j < a.length; ++j) {
      d += (a[j] - b[j]) * (a[j] - b[j])
    }
    distance += Math.sqrt(d) * features[i].importance
  }

  return distance;
}

const INF = Number.MAX_SAFE_INTEGER;
function tsp(costMatrix, n, range) {
  const dp = Array.from({ length: 1 << n }, () => Array(n).fill(INF));
  const log = []
  for (let i = 0; i < n; ++i) {
    log[1 << i] = i;
  }
  dp[1][0] = 0

  for (let mask = 1; mask < (1 << n); mask++) {
    for (let p = mask; p != 0; p -= (p & -p)) {
      let u = log[p & -p];
      for (let q = (1 << n) - 1 - mask; q != 0; q -= (q & -q)) {
        let v = log[q & -q];
        if (dp[mask][u] + costMatrix[u][v] < dp[mask | (1 << v)][v]) {
          dp[mask | (1 << v)][v] = dp[mask][u] + costMatrix[u][v];
        }
      }
    }
  }

  // Find the minimum cost to return to the starting city
  let minCost = INF, end = 0
  for (let u = 1; u < n; u++) {
    if (dp[(1 << n) - 1][u] < minCost) {
      minCost = dp[(1 << n) - 1][u];
      end = u
    }
  }

  return minCost;
}

function calculateTotalDistance(dist, path) {
  let totalDistance = 0;
  let alpha = 0.8, rate = 1;
  for (let k = 1; k < 7; ++k) {
    for (let i = 0; i < path.length - k; i++) {
      totalDistance += dist[path[i]][path[i + k]] * rate;
    }
    rate *= alpha;
  }
  return totalDistance;
}

function calculateTotalDistanceNew(dists, importances, path) {
  let totalDistance = 0
  let alpha = 0.9999
  for (let j = 0; j < importances.length; ++j) {
    let dist = dists[j]
    let rate = alpha
    let featureDistance = 0
    let d = []
    for (let k = 1; k < 2; ++k) {
      for (let i = 0; i < path.length - k; i++) {
        // d.push(dist[path[i]][path[i + k]])
        featureDistance += dist[path[i]][path[i + k]] * rate;
      }
      rate *= alpha
    }

    totalDistance += featureDistance * importances[j]
  }
  return totalDistance
}

function swap(arr, i, j) {
  let temp = arr[i]
  arr[i] = arr[j]
  arr[j] = temp
}

function insert(arr, i, j) {
  let temp = arr[j]
  for (let k = j; k > i; --k) {
    arr[k] = arr[k - 1]
  }
  arr[i] = temp
}

function adjustList(items, range, state) {
  let temperature = 200
  const coolingRate = 0.9999
  const numItems = items.length
  const windowSize = 12

  let currentPath = Array.from({ length: numItems }, (_, index) => index);
  let dist = [], feature_dist = [], feature_importances = [], n_features = 0
  const min_importance = state.data_features.map(d => d.importance).sort((a, b) => b - a)[10]
  for (let i = 0; i < items.length; ++i) {
    let row = []
    for (let j = 0; j < items.length; ++j) {
      row.push(calculateDistance(items[i], items[j], state.data_features, min_importance));
    }
    dist.push(row);
  }
  for (let k = 0; k < state.data_features.length; ++k) {
    if (state.data_features[k].importance >= min_importance) {
      let mat = []
      for (let i = 0; i < items.length; ++i) {
        let row = []
        for (let j = 0; j < items.length; ++j) {
          row.push(calculateFeatureDistance(items[i], items[j], k, state.data_features[k].default));
        }
        mat.push(row)
      }
      let importance = state.data_features[k].importance
      if (state.data_features[k].name == 'Income' || state.data_features[k].name == 'CreditScore') {
        importance *= 1
      }
      console.log(state.data_features[k].name, importance)
      feature_dist.push(mat)
      feature_importances.push(importance)
      n_features++
    }
  }

  let bestPath = [...currentPath];
  let currentDistance = calculateTotalDistanceNew(feature_dist, feature_importances, currentPath) // calculateTotalDistance(dist, currentPath);
  let bestDistance = currentDistance;

  const drng = new RandomGenerator(5)

  while (temperature > 1e-3) {
    let newPath = [...currentPath];
    let i = Math.floor(drng.random() * (numItems - windowSize));
    let j = i + Math.ceil(drng.random() * windowSize);
    if (j > numItems) {
      continue
    }

    if (drng.random() < 0.8) {
      swap(newPath, i, j)
    } else {
      insert(newPath, i, j)
    }

    let flag = 0
    for (let k = i; k <= j; ++k) {
      if (k < range[newPath[k]][0] || k > range[newPath[k]][1]) {
        flag = 1
      }
    }
    if (flag) {
      continue
    }

    let newDistance = calculateTotalDistanceNew(feature_dist, feature_importances, newPath) //calculateTotalDistance(dist, newPath);
    if (newDistance < currentDistance || Math.exp((currentDistance - newDistance) / temperature) > drng.random()) {
      //console.log('newDistance', newDistance, 'currentDistance', currentDistance)
      currentPath = [...newPath];
      currentDistance = newDistance;

      if (currentDistance < bestDistance) {
        bestPath = [...currentPath];
        bestDistance = currentDistance;
      }
    }

    temperature *= coolingRate;
  }
  const pairwiseDist = []
  for (let i = 0; i < 15; ++i) {
    const d = []
    for (let j = 0; j < 15; ++j) {
      d.push(dist[bestPath[i]][bestPath[j]])
    }
    pairwiseDist.push(d)
  }

  return { distance: bestDistance, path: bestPath };
}

export function sortList(items, zoom_level, state, adjust = 1, threshold = 0.1) {
  const order_keys = JSON.parse(JSON.stringify(state.matrixview.order_keys))
  const preserved_keys = new Set(state.matrixview.extended_cols.map(d => d.index))
  let threshold_ = threshold
  threshold = 0

  let cmp = (a, b) => 0
  if (order_keys.length == 0) {
    // If there are no order keys and the zoom level is 0, sort by weight or coverage
    if (zoom_level == 0) {
      if (state.model_info.weighted) {
        order_keys.push({ key: 'weight', order: -1 })
      } else {
        order_keys.push({ key: 'coverage', order: -1 })
      }
    }
  }
  cmp = (a, b) => {
    for (let index = 0; index < order_keys.length; ++index) {
      const key = order_keys[index].key;
      const order = order_keys[index].order;
      const by_value = preserved_keys.has(key)
      if (by_value) {
        if (adjust && Math.abs(a[key] - b[key]) < Math.min(a[key], b[key]) * threshold) {
          return 0
        } else {
          return order * (a[key] - b[key]);
        }
      } else {
        if (!a.cond_dict[key] && b.cond_dict[key]) {
          return 1;
        } else if (a.cond_dict[key] && !b.cond_dict[key]) {
          return -1;
        } else if (!a.cond_dict[key] && !b.cond_dict[key]) {
          continue;
        } else if (a.range_key[key] != b.range_key[key]) {
          if (typeof (a.range_key[key]) == 'string' && a.range_key[key][0] == '1' && b.range_key[key][0] == '1') {
            return -order * (a.range_key[key] - b.range_key[key]);
          } else {
            if (adjust && Math.abs(a.cond_norm[key][0] - b.cond_norm[key][0]) < Math.min(a.cond_norm[key][0], b.cond_norm[key][0]) * threshold &&
              Math.abs(a.cond_norm[key][1] - b.cond_norm[key][1]) < Math.min(a.cond_norm[key][1], b.cond_norm[key][1]) * threshold) {
              return 0
            } else {
              return +order * (a.range_key[key] - b.range_key[key])
            }
          }
        }
      }
    }
    return a.predict - b.predict;
  }

  let current_cmp
  if (zoom_level != 0) {
    // If the zoom level is not 0, handle unique representatives and sort within groups
    const unique_reps = [...new Set(items.map(d => d.father).filter(d => d != -1))];
    const rep_order = {}

    for (let i = 0; i < unique_reps.length; ++i) {
      rep_order[unique_reps[i]] = i
    }
    rep_order[-1] = -1

    let new_cmp = (a, b) => {
      if (rep_order[a.father] != rep_order[b.father]) {
        return rep_order[a.father] - rep_order[b.father]
      } else if (a.represent != b.represent) {
        return b.represent - a.represent
      } else {
        return cmp(a, b)
      }
    }
    current_cmp = new_cmp
  } else {
    current_cmp = cmp
  }
  items = items.sort(current_cmp)

  if (adjust) {
    const range = []
    threshold = threshold_
    for (let i = 0; i < items.length; ++i) {
      let left = i, right = i
      while (left > 0 && current_cmp(items[left - 1], items[i]) == 0) --left
      while (right + 1 < items.length && current_cmp(items[i], items[right + 1]) == 0) ++right
      range.push([left, right])
    }

    const ret = adjustList(items, range, state)
    // swap(ret.path, 7, 9)
    // swap(ret.path, 8, 10)
    items = ret.path.map(d => items[d])
  }
  return items
}

