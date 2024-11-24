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

function generateRandomDistanceMatrix(n) {
  let matrix = [];
  for (let i = 0; i < n; i++) {
      matrix[i] = [];
      for (let j = 0; j < n; j++) {
          if (j < i) {
              matrix[i][j] = matrix[j][i];
          } else if (i === j) {
              matrix[i][j] = 0;
          } else {
              matrix[i][j] = Math.floor(Math.random() * 100) + 1;  // Random distances between 1 and 100
          }
      }
  }
  return matrix;
}

function countInversions(arr) {
  let count = 0;
  const n = arr.length;

  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      if (arr[i] > arr[j]) {
        count++;
      }
    }
  }

  return count;
}

const herdKarpN = 10

function compareInversions(a, b) {
  const inversionsA = countInversions([].concat(...a));
  const inversionsB = countInversions([].concat(...b));

  if (inversionsA <= inversionsB) {
    return a;
  } else {
    return b;
  }
}

function randint(n) {
  return Math.min(Math.floor(Math.random() * n), n - 1)
}

const maxCost = 100

class UnionFind {
  constructor(size) {
      this.parent = Array.from({ length: size }, (_, index) => index);
      this.rank = new Array(size).fill(0);
      this.size = new Array(size).fill(1); // Track size of each component
      this.component = Array.from({ length: size }, (_, index) => [index]); // Track nodes in each component
  }

  find(x) {
      if (this.parent[x] !== x) {
          this.parent[x] = this.find(this.parent[x]);
      }
      return this.parent[x];
  }

  union(x, y) {
      let rootX = this.find(x);
      let rootY = this.find(y);
      if (rootX !== rootY) {
          if (this.rank[rootX] > this.rank[rootY]) {
              this.parent[rootY] = rootX;
              this.size[rootX] += this.size[rootY];
              this.component[rootX] = this.component[rootX].concat(this.component[rootY]);
          } else if (this.rank[rootX] < this.rank[rootY]) {
              this.parent[rootX] = rootY;
              this.size[rootY] += this.size[rootX];
              this.component[rootY] = this.component[rootY].concat(this.component[rootX]);
          } else {
              this.parent[rootY] = rootX;
              this.rank[rootX]++;
              this.size[rootX] += this.size[rootY];
              this.component[rootX] = this.component[rootX].concat(this.component[rootY]);
          }
      }
  }
}

function extractSubgraph(adj, V) {
  const newAdj = Array(V.length).fill(null).map(() => Array(V.length).fill(Infinity))

  for (let i = 0; i < V.length; ++i) {
    for (let j = 0; j < V.length; ++j) {
      newAdj[i][j] = adj[V[i]][V[j]]
    }
  }
  return newAdj
}

function calcPathCost(path, cost) {
  let ret = 0
  for (let i = 0; i < path.length - 1; ++i) {
    ret += cost[path[i]][path[i + 1]]
  }
  return ret
}

function solveSmallTSP(cost) {
  if (cost.length <= 1) {
    return [{ path: [0], cost: 0 }]
  }

  const n = cost.length
  const ret = []
  for (let start = 0; start < n; ++start) {
    const dp = new Array(1 << n).fill(null).map(() => new Array(n).fill(Infinity))
    dp[1 << start][start] = 0
    for (let mask = 1; mask < (1 << n); mask++) if (mask & (1 << start)) {
        for (let i = 0; i < n; i++) {
            if ((mask & (1 << i)) === 0) continue
            for (let j = 0; j < n; j++) if (cost[i][j] < Infinity) {
                if ((mask & (1 << j)) === 0 || i === j) continue
                const prevMask = mask ^ (1 << j)
                dp[mask][j] = Math.min(dp[mask][j], dp[prevMask][i] + cost[i][j])
            }
        }
    }
    for (let end = 0; end < n; ++end) if (end != start) {
      let i, j = end, mask = (1 << n) - 1
      if (dp[mask][end] == Infinity) {
        continue
      }
      let path = [j]
      while (mask) {
        const prevMask = mask ^ (1 << j)
        for (i = 0; i < n; ++i) {
          if ((prevMask & (1 << i)) && dp[prevMask][i] + cost[i][j] == dp[mask][j]) {
            break
          }
        }
        if (i < n) {
          path.push(i)
        }
        mask = prevMask
        j = i
      }
      path = path.reverse()
      ret.push({ path, cost: calcPathCost(path, cost) })
    }
  }

  return ret
}

function clusterGraph(cost, maxComponentSize, preservedGroups = []) {
  const nodes = cost.length
  const edges = []
  const uf = new UnionFind(nodes)
  const preservedNodes = new Set([].concat(...preservedGroups))

  for (let i = 0; i < nodes; i++) if (!preservedNodes.has(i)) {
      for (let j = i + 1; j < nodes; j++) if (!preservedNodes.has(j)) {
          if (cost[i][j] >= 0 && cost[i][j] < Infinity) {
              edges.push([cost[i][j], i, j])
          }
      }
  }

  edges.sort((a, b) => a[0] - b[0])

  for (const [weight, start, end] of edges) {
      if (uf.find(start) !== uf.find(end)) {
          if (uf.size[uf.find(start)] + uf.size[uf.find(end)] <= maxComponentSize) {
              uf.union(start, end)
          }
      }
  }

  for (let group of preservedGroups) {
    for (let i = 0; i < group.length - 1; ++i) {
      uf.union(group[i], group[i + 1])
    }
  }

  // Collect unique components
  const components = {}
  for (let i = 0; i < nodes; i++) {
      const root = uf.find(i);
      if (!components[root]) {
          components[root] = uf.component[root]
      }
  }

  return Object.values(components)
}

function solveTSP(cost) {
  if (cost.length <= herdKarpN) {
    return solveSmallTSP(cost)
  }
  
  const components = clusterGraph(cost, herdKarpN)
  const componentId = {}
  const componentPath = []
  const componentCost = []
  console.log('components', components)
  for (let i = 0; i < cost.length; ++i) {
    componentPath.push({})
    componentCost.push({})
  }

  components.forEach((comp, index) => {
    for (let j of comp) {
      componentId[j] = index
    }
    const subG = extractSubgraph(cost, comp)
    const paths = solveSmallTSP(subG)
    paths.forEach(d => d.path = d.path.map(i => comp[i]))
    paths.forEach(d => {
      const start = d.path[0]
      const end = d.path[d.path.length - 1]
      if (!componentPath[start]) {
        componentPath[start] = {}
        componentCost[start] = {}
      }
      componentPath[start][end] = d.path
      componentCost[start][end] = d.cost
    })
  })
  const n = components.length
  const m = cost.length

  for (let i = 0; i < m; ++i) {
    for (let j = 0; j < m; ++j) if (componentId[i] != componentId[j]) {
      for (let k of components[componentId[j]]) if (k != j || components[componentId[j]].length == 1) {
        if (!componentCost[i]) {
          componentCost[i] = {}
          componentPath[i] = {}
        }
        if (!componentCost[i][j] || componentCost[i][j] > cost[i][k] + componentCost[k][j])
        componentPath[i][j] = componentPath[k][j]
        componentCost[i][j] = cost[i][k] + componentCost[k][j]
      }
    }
  }

  const dp = new Array(1 << n).fill(null).map(() => new Array(m).fill(Infinity))

  for (let j = 0; j < m; ++j) {
    const cj = componentId[j]
    for (let i of components[componentId[j]]) {
      if (componentPath[i][j])
      if (componentCost[i][j] < dp[1 << cj][j]) {
        dp[1 << cj][j] = componentCost[i][j]
      }
    }
    dp[0][j] = 0
  }
  for (let mask = 1; mask < (1 << n); mask++) {
    for (let i = 0; i < m; i++) {
      const ci = componentId[i]
      if ((mask & (1 << ci)) === 0) continue
      for (let j = 0; j < m; j++) {
        const cj = componentId[j]
        if ((mask & (1 << cj)) === 0 || ci == cj) continue
        const prevMask = mask ^ (1 << cj)
        if (componentPath[i][j]) {
          dp[mask][j] = Math.min(dp[mask][j], dp[prevMask][i] + componentCost[i][j])
        }
      }
    }
  }

  const ret = []
  for (let end = 0; end < m; ++end) {
    let i, j = end, mask = (1 << n) - 1
    let path = []
    while (mask) {
      const prevMask = mask ^ (1 << componentId[j])
      for (i = 0; i < m; ++i) {
        if (componentPath[i][j])
        if ((prevMask & (1 << componentId[i])) && dp[prevMask][i] + componentCost[i][j] == dp[mask][j]) {
          break
        }
      }
      if (i < m) {
        const sub = componentPath[i][j]
        for (let k = sub.length - 1; k >= 0; --k) {
          path.push(sub[k])
        }
      } else {
        i = -1
        for (let k of components[componentId[j]]) if (componentPath[k][j]) {
          if (i == -1 || componentCost[k][j] < componentCost[i][j]) {
            i = k
          }
        }
        if (i == -1) {
          break
        }
        const sub = componentPath[i][j]
        for (let k = sub.length - 1; k >= 0; --k) {
          path.push(sub[k])
        }
      }
      mask = prevMask
      j = i
    }
    path = path.reverse()
    if (path.length == cost.length) {
      ret.push({ path, cost: calcPathCost(path, cost) })
    }
  }

  return ret
}

function solveBlock(block, cost) {  
  const n = cost.length
  const dp = new Array(n).fill(Infinity)
  const pre = new Array(n)
  const currentPath = new Array(n)
  const startPath = {}
  const startCost = {}
  for (let i of block[0]) {
    startPath[i] = {}
    startCost[i] = {}
  }

  for (let k = 0; k < block.length; ++k) {
    const comp = block[k]
    let subG, paths
    if (comp.length > 1) {
      subG = extractSubgraph(cost, comp)
      paths = solveTSP(subG)
      paths.forEach(d => d.path = d.path.map(i => comp[i]))
    } else {
      paths = [{ path: [comp[0]], cost: 0 }]
    }

    if (block.length == 1) {
      return paths
    }

    for (let p of paths) {
      const start = p.path[0]
      const end = p.path[p.path.length - 1]
      if (k == 0) {
        startPath[start][end] = p.path
        startCost[start][end] = p.cost
        if (p.cost < dp[end]) {
          dp[end] = p.cost
          pre[end] = -1
          currentPath[end] = p.path
        }
      } else {
        for (let i of block[k - 1]) if (dp[i] < Infinity) {
          if (dp[i] + cost[i][start] + p.cost < dp[end]) {
            dp[end] = dp[i] + cost[i][start] + p.cost
            pre[end] = i
            currentPath[end] = p.path
          }
        }
      }
    }
  }

  const ret = []
  for (let end of block[block.length - 1]) if (dp[end] < Infinity) {
    let path = []
    let i = end
    while (i != -1 && pre[i] != -1) {
      for (let j = currentPath[i].length - 1; j >= 0; --j) {
        path.push(currentPath[i][j])
      }
      i = pre[i]
    }
    for (let j of block[0]) if (startPath[j][i]) {
      let newPath = path.concat(startPath[j][i])
      ret.push({ path: newPath, cost: calcPathCost(newPath, cost) })
      newPath = newPath.slice(0).reverse()
      ret.push({ path: newPath, cost: calcPathCost(newPath, cost) })
    }
  }
  return ret
}

function findPathTSP(adjacentBlocks, cost) {
  if (adjacentBlocks.length == 0) {
    return solveTSP(cost)
  }

  const m = cost.length
  const preservedGroups = adjacentBlocks.map(block => [].concat(...block))
  const blockId = new Array(m).fill(-1)
  preservedGroups.forEach((group, i) => {
    for (let j of group) {
      blockId[j] = i
    }
  })
  
  const components = clusterGraph(cost, herdKarpN, preservedGroups)
  const componentId = {}
  const componentPath = []
  const componentCost = []
  const n = components.length
  if (n == 1) {
    return solveBlock(adjacentBlocks[0], cost)
  }

  for (let i = 0; i < m; ++i) {
    componentPath.push({})
    componentCost.push({})
  }
  
  components.forEach((comp, index) => {
    for (let j of comp) {
      componentId[j] = index
    }
    let subG = extractSubgraph(cost, comp)
    let paths
    const lastTime = new Date()
    if (blockId[comp[0]] == -1) {
      paths = solveSmallTSP(subG)
    } else {
      const nodeId = {}
      for (let i = 0; i < comp.length; ++i) {
        nodeId[comp[i]] = i
      }
      const block = adjacentBlocks[blockId[comp[0]]].map(sub => sub.map(i => nodeId[i]))
      paths = solveBlock(block, subG)
    }
    const currentTime = new Date()
    const time = currentTime - lastTime
    if (time > 100) {
      console.log('time', index, time, comp, subG, blockId[comp[0]] == -1)
    }
    paths.forEach(d => d.path = d.path.map(i => comp[i]))
    paths.forEach(d => {
      const start = d.path[0]
      const end = d.path[d.path.length - 1]
      componentPath[start][end] = d.path
      componentCost[start][end] = d.cost
    })
  })
  console.log('number of components', n)

  for (let i = 0; i < m; ++i) {
    for (let j = 0; j < m; ++j) if (componentId[i] != componentId[j]) {
      for (let k of components[componentId[j]]) {
        if (!componentCost[i][j] || (componentPath[k][j] && componentCost[i][j] > cost[i][k] + componentCost[k][j])) {
          componentPath[i][j] = componentPath[k][j]
          componentCost[i][j] = cost[i][k] + componentCost[k][j]
        }
      }
    }
  }

  const dp = new Array(1 << n).fill(null).map(() => new Array(m).fill(Infinity))

  for (let j = 0; j < m; ++j) {
    const cj = componentId[j]
    for (let i of components[componentId[j]]) {
      if (componentCost[i][j] < dp[1 << cj][j]) {
        dp[1 << cj][j] = componentCost[i][j]
      }
    }
    dp[0][j] = 0
  }
  for (let mask = 1; mask < (1 << n); mask++) {
    for (let i = 0; i < m; i++) {
      const ci = componentId[i]
      if ((mask & (1 << ci)) === 0) continue
      for (let j = 0; j < m; j++) {
        const cj = componentId[j]
        if ((mask & (1 << cj)) === 0) continue
        const prevMask = mask ^ (1 << cj)
        if (componentPath[i][j]) {
          dp[mask][j] = Math.min(dp[mask][j], dp[prevMask][i] + componentCost[i][j])
        }
      }
    }
  }

  const ret = []
  let i, j, mask
  for (let end = 0; end < m; ++end) {
    j = end
    mask = (1 << n) - 1
    if (dp[mask][end] == Infinity) {
      continue
    }
    let path = []
    while (mask) {
      const prevMask = mask ^ (1 << componentId[j])
      for (i = 0; i < m; ++i) {      
        if (componentPath[i][j])
        if ((prevMask & (1 << componentId[i])) && dp[prevMask][i] + componentCost[i][j] == dp[mask][j]) {
          break
        }
      }
      if (i < m) {
        const sub = componentPath[i][j]
        for (let k = sub.length - 1; k >= 0; --k) {
          path.push(sub[k])
        }
      } else {
        i = -1
        for (let k of components[componentId[j]]) if (k != j && componentPath[k][j]) {
          if (i == -1 || componentCost[k][j] < componentCost[i][j]) {
            i = k
          }
        }
        if (i == -1 || !componentPath[i][j]) {
          break
        }
        const sub = componentPath[i][j]
        for (let k = sub.length - 1; k >= 0; --k) {
          path.push(sub[k])
        }
      }
      mask = prevMask
      j = i
    }
    path = path.reverse()
    ret.push({ path, cost: calcPathCost(path, cost) })
  }
  return ret
}

function chooseMinCostPath(paths, isFirst) {
  let minCost = Infinity
  let ret
  if (isFirst) {
    const minStart = Math.min(...paths.map(d => d.path[0]))
    paths = paths.filter(d => d.path[0] == minStart)
  }
  for (let p of paths) {
    if (p.cost < minCost) {
      minCost = p.cost
      ret = p
    }
  }
  return [ret.path, ret.cost]
}

function findPathTrail(adjacent_blocks, adjacent, skip_adajcent, max_skips = 0) {
  let indeg = Array.from({ length: adjacent.length }, () => 0)
  adjacent.forEach(adj => adj.forEach(d => indeg[d]++))
  let visited = new Set()

  let s
  let candidates = []
  if (adjacent_blocks.length == 0) {
    for (let i = 0; i < indeg.length; ++i) {
      if (indeg[i] == 0) {
        candidates.push(i)
      }
    }
  }

  const in_blocks = new Set([].concat(...[].concat(...adjacent_blocks)))
  const in_block_starts = new Set([].concat(...adjacent_blocks.map(blocks => blocks[0])))
  if (candidates.length > 0) {
    s = candidates[randint(candidates.length)]
  } else {
    candidates = Array.from({ length: adjacent.length }, (_, i) => i).filter(i => !in_blocks.has(i))
    if (candidates.length > 0) {
      s = candidates[randint(candidates.length)]
    } else {
      candidates = [...in_block_starts]
      if (candidates.length > 0) {
        s = candidates[randint(candidates.length)]
      }
    }
  }

  let path = [s]
  let skips = 0
  let current_block, pos, index
  if (in_blocks.has(s)) {
    for (let block of adjacent_blocks) {
      if (block[0].indexOf(s) != -1) {
        current_block = block
        break
      }
    }
    pos = 0
    index = 0
  }
  // console.log(s, adjacent_blocks, adjacent)

  while (path.length < adjacent.length) {
    visited.add(s)
    let neighbors
    let visiting_block = in_blocks.has(s)

    if (visiting_block) {
      pos += 1
      if (pos == current_block[index].length) {
        pos = 0
        index += 1
      }
    }

    if (!visiting_block || index == current_block.length) {
      neighbors = adjacent[s].filter(i => !visited.has(i) && (in_block_starts.has(i) || !in_blocks.has(i)))
    } else {
      neighbors = adjacent[s].filter(i => !visited.has(i) && current_block[index].indexOf(i) != -1)
    }

    if (neighbors.length == 0) {
      skips += 1
      if (skips > max_skips) {
        break
      }

      if (!visiting_block || index == current_block.length) {
        neighbors = skip_adajcent[s].filter(i => !visited.has(i) && (in_block_starts.has(i) || !in_blocks.has(i)))
      } else {
        neighbors = skip_adajcent[s].filter(i => !visited.has(i) && current_block[index].indexOf(i) != -1)
      }

      if (neighbors.length == 0) {
        break
      }
    }

    const t = neighbors[randint(neighbors.length)]
    path.push(t)
    s = t
    if ((!visiting_block || index == current_block.length) && in_blocks.has(s)) {
      for (let block of adjacent_blocks) {
        if (block[0].indexOf(s) != -1) {
          current_block = block
          break
        }
      }
      pos = 0
      index = 0
    }
  }

  return [path, skips]
}

function findPath(blocks, adjacent, skip_adajcent) {
  let ret = []
  let min_skips = adjacent.length

  for (let k = 0; k < 1000; ++k) {
    const [current, skips] = findPathTrail(blocks, adjacent, skip_adajcent, min_skips)
    if (skips < min_skips && current.length == adjacent.length) {
      min_skips = skips
      ret = current
    } else if (current.length > ret.length) {
      min_skips = skips
      ret = current
    }
    if (min_skips == 0) {
      break
    }
  }
  return [ret, min_skips]
}

function featureOptimize(items, blocks, adjacent, costMatrix, feature, threshold, isFirst) {
  const feature_index = feature.index
  const key = feature_index

  const adjacent_item = {}
  let ordered_items = []
  if (feature.dtype != 'category') {
    ordered_items = items.filter(d => d.cond_norm[feature_index])
    .sort((a, b) => {
      if (!a.cond_dict[key] && b.cond_dict[key]) {
        return 1
      } else if (a.cond_dict[key] && !b.cond_dict[key]) {
        return -1
      } else if (!a.cond_dict[key] && !b.cond_dict[key]) {
        return 0
      } 
      if (a.range_key[key] != b.range_key[key]) {
        if (typeof (a.range_key[key]) == 'string' && a.range_key[key][0] == '1' && b.range_key[key][0] == '1') {
          return a.range_key[key] - b.range_key[key];
        } else {
          if (a.cond_norm[key][0] == b.cond_norm[key][0] && a.cond_norm[key][1] == b.cond_norm[key][1]) {
            return 0
          } else {
            return a.range_key[key] - b.range_key[key]
          }
        }
      } else {
        return 0
      }
    })
  
    for (let i = 0; i < ordered_items.length - 1; ++i) {
      adjacent_item[ordered_items[i].id] = ordered_items[i + 1].id
    }
  }

  function featureAdjacent(point1, point2, feature_index, threshold) {
    if (!point1.cond_norm[feature_index] || !point2.cond_norm[feature_index] || point1.predict != point2.predict) {
      return 0
    }
    
    const a = point1.cond_norm[feature_index]
    const b = point2.cond_norm[feature_index]
    if ((adjacent_item[point1.id] == point2.id || adjacent_item[point2.id] == point1.id) && (
        Math.abs(a[0] - b[0]) < threshold || Math.abs(a[1] - b[1]) < threshold)) {
      return 1
    }

    if (a.length > 2) {
      threshold = 1e-4
    }
    for (let j = 0; j < a.length; ++j) {
      if (Math.abs(a[j] - b[j]) > threshold) {
        return -1
      }
    }
    return 1
  }

  function featureDist(point1, point2, feature_index) {
    if (!point1.cond_norm[feature_index] || !point2.cond_norm[feature_index] || point1.predict != point2.predict) {
      return 1
    }
    
    const a = point1.cond_norm[feature_index]
    const b = point2.cond_norm[feature_index]
    if ((adjacent_item[point1.id] == point2.id || adjacent_item[point2.id] == point1.id) && (a[0] == b[0] || a[1] == b[1])) {
      return 0
    }

    let dist = 0
    for (let j = 0; j < a.length; ++j) {
      const d = a[j] - b[j]
      dist += d * d
    }
    return Math.sqrt(dist)
  }

  /*
  const skip_adjacent = adjacent.map((adj, i) => adj.filter(j =>
    items[i].cond_norm[feature_index] && featureAdjacent(items[i], items[j], feature_index, threshold) != 1)
  )
  const remain_adjacent = adjacent.map((adj, i) => adj.filter(j =>
    !items[i].cond_norm[feature_index] || featureAdjacent(items[i], items[j], feature_index, threshold) == 1)
  )
  */
  for (let i = 0; i < costMatrix.length; ++i) {
    costMatrix[i][i] = 0
  } 

  const costMatrix2 = JSON.parse(JSON.stringify(costMatrix))
  const relations = []
  adjacent.forEach((adj, i) => {
    const current = []
    adj.forEach(j => {
      if (items[i].cond_norm[feature_index] && items[j].cond_norm[feature_index]) {
        current.push([items[i].cond_norm[feature_index], items[j].cond_norm[feature_index], featureAdjacent(items[i], items[j], feature_index, threshold)])
        if (featureAdjacent(items[i], items[j], feature_index, threshold) != 1) {
          costMatrix2[i][j] = (featureDist(items[i], items[j], feature_index)) * feature.importance
          costMatrix[i][j] += (featureDist(items[i], items[j], feature_index)) * feature.importance
        } else {
          costMatrix2[i][j] = 0
        }
      } else if (items[i].cond_norm[feature_index] || items[j].cond_norm[feature_index]) {
        costMatrix[i][j] += feature.importance
        costMatrix2[i][j] += feature.importance
      }
    })
    if (items[i].cond_norm[feature_index]) {
      relations.push(current)
    }
  })
  
  const [path, min_skips] = chooseMinCostPath(findPathTSP(blocks, costMatrix2), isFirst)
  // const [path, min_skips] = findPath(blocks, remain_adjacent, skip_adjacent)

  let current_block = []
  let sizes = []
  let conds = path.filter(i => items[i].cond_norm[feature_index]).map(i => items[i].cond_norm[feature_index])
  for (let i = 0; i < path.length; ++i) {
    let u = path[i]
    if (!items[u].cond_norm[feature_index]) {
      continue
    } else {
      current_block.push(u)
    }
    if (i == path.length - 1 || featureAdjacent(items[u], items[path[i + 1]], feature_index, threshold * 3) != 1) {
      if (current_block.length > 1) {
        blocks.push([current_block])
        sizes.push(current_block.length)
      }
      for (let j of current_block) {
        adjacent[j].forEach(k => costMatrix[j][k] += maxCost)
        adjacent[j] = adjacent[j].filter(k => current_block.indexOf(k) == -1 || featureAdjacent(items[j], items[k], feature_index, threshold) != -1)
        adjacent[j].forEach(k => costMatrix[j][k] -= maxCost)
      }
      current_block = []
    }
  }
  console.log('feature', feature.name, 'sizes', sizes, 'conds', conds, 'relations', relations)

  const block_ranges = []
  for (let block of blocks) {
    for (let sub of block) {
      for (let i = 0; i < path.length; ++i) {
        if (sub.indexOf(path[i]) != -1) {
          block_ranges.push([i, i + sub.length])
          break
        }
      }
    }
  }

  const block_events = block_ranges.map(r => [r[0], 0])
    .concat(block_ranges.map(r => [r[1], 1]))
    .sort((a, b) => a[0] != b[0] ? a[0] - b[0] : a[1] - b[1])
  
  while (blocks.length > 0) {
    blocks.pop()
  }

  let cnt = 0, last, new_block = []
  for (let ev of block_events) {
    if (cnt > 0 && last < ev[0]) {
      let current = []
      for (let i = last; i < ev[0]; ++i) {
        current.push(path[i])
      }
      new_block.push(current)
      last = ev[0]
    } else if (cnt == 0) {
      last = ev[0]
    }

    if (ev[1] == 0) {
      cnt += 1
    } else {
      cnt -= 1
    }

    if (cnt == 0) {
      if (new_block.length == 1) {
        blocks.push(new_block)
      } else {
        blocks.push(new_block)//compareInversions(new_block, new_block.slice(0).reverse()))
        //blocks.push(new_block.slice(0).reverse())
      }
      new_block = []
    }
  }

  const blockId = {};
  [].concat(...new_block).forEach((block, i) => {
    for (let j of block) {
      blockId[j] = i
    }
  })

  adjacent.forEach((adj, i) => {
    adj.forEach(j => {
      if (items[i].cond_norm[feature_index] && items[j].cond_norm[feature_index]) {
        if (blockId[i] == blockId[j] && featureAdjacent(items[i], items[j], feature_index, threshold) != 1) {
          costMatrix[i][j] -= feature.importance
        }
      } else if (items[i].cond_norm[feature_index] || items[j].cond_norm[feature_index]) {
        //costMatrix[i][j] -= feature.importance
      }
    })
  })

  return [path, adjacent, costMatrix]
}

function adjustList(items, adjacent, state, threshold, isFirst = 0) {
  const keys = state.matrixview.order_keys.map(d => d.key)
  const ordered_features = state.data_features.slice(0)
    .sort((a, b) => b.importance - a.importance)
    .filter(d => keys.indexOf(d.index) == -1)

  const drng = new RandomGenerator(5)
  let path = []
  let blocks = []
  let costMatrix = Array(adjacent.length).fill(null).map(() => Array(adjacent.length).fill(maxCost))
  adjacent.forEach((adj, i) => {
    adj.forEach(j => {
      costMatrix[i][j] = 0
    })
  })

  for (let feature of ordered_features.slice(0, 10)) {
    [path, adjacent, costMatrix] = featureOptimize(items, blocks, adjacent, costMatrix, feature, threshold, isFirst)
  }
  // console.log(path)

  return { path: path }
}

export function sortList(items, zoom_level, state, adjust = 1, threshold = 0.15) {
  const order_keys = JSON.parse(JSON.stringify(state.matrixview.order_keys))
  const noPreOrder = order_keys.length == 0
  const preserved_keys = new Set(state.matrixview.extended_cols.map(d => d.index))
  let threshold_ = threshold
  threshold = 0
  if (order_keys.length && order_keys[0].key == 'anomaly') {
    adjust = 0
  }

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
            if (adjust && Math.abs(a.cond_sort[key][0] - b.cond_sort[key][0]) < threshold &&
              Math.abs(a.cond_sort[key][1] - b.cond_sort[key][1]) < threshold) {
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
    let adjacent = [], new_items = []
    const initial_position = {}, new_position = {}
    for (let i = 0; i < items.length; ++i) {
      initial_position[items[i].id] = i
    }

    threshold = threshold_
    for (let i = 0, last = 0, max_right = 0; i < items.length; ++i) {
      let left = i, right = i, current_adjacent = []
      while (left > last && current_cmp(items[left - 1], items[i]) == 0) {
        left -= 1
        current_adjacent.push(left - last)
      }
      while (right + 1 < items.length && current_cmp(items[i], items[right + 1]) == 0) {
        right += 1
        current_adjacent.push(right - last)
      }
      current_adjacent = current_adjacent.sort()
      adjacent.push(current_adjacent)
      if (right > max_right) {
        max_right = right
      }

      if (max_right == i || i == items.length - 1) {
        const isFirst = last == 0 && noPreOrder
        console.log('solve', last, right + 1)
        let ret, sub_items
        if (noPreOrder && last > 0) {
          const lastItem = new_items[new_items.length - 1]
          for (let j = 0; j < adjacent.length; ++j) {
            adjacent[j] = adjacent[j].map(d => d + 1)
          }
          adjacent = [[]].concat(adjacent)
          for (let j = 1; j < adjacent.length; ++j) if (current_cmp(lastItem, items[j + last - 1]) == 0) {
            adjacent[0].push(j)
          }
          ret = adjustList([lastItem].concat(items.slice(last, right + 1)), adjacent, state, threshold, 1)
          sub_items = ret.path.slice(1).map(d => items[d + last - 1])
        } else {
          ret = adjustList(items.slice(last, right + 1), adjacent, state, threshold, isFirst)
          sub_items = ret.path.map(d => items[d + last])
        }
        
        for (let j = 0; j < sub_items.length; ++j) {
          new_position[sub_items[j].id] = j
          if (j == 0 || sub_items[j].predict != sub_items[j - 1].predict) {
            let k = j
            while (k + 1 < sub_items.length && sub_items[k + 1].predict == sub_items[j].predict) {
              ++k
            }
            let avg_position = 0
            for (let l = j; l <= k; ++l) {
              avg_position += initial_position[sub_items[l].id]
            }
            avg_position = avg_position / (k - j + 1)
            for (let l = j; l <= k; ++l) {
              initial_position[sub_items[l].id] = avg_position
            }
          }
        }
        sub_items = sub_items.sort((a, b) => {
          if (initial_position[a.id] != initial_position[b.id]) {
            return initial_position[a.id] - initial_position[b.id]
          } else {
            return new_position[a.id] != new_position[b.id]
          }
        })

        new_items = new_items.concat(sub_items)
        last = right + 1
        adjacent = []
      }
    }
    items = new_items
  }
  return items
}
