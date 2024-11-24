// util functions
var map_to_request_params = function(kwargs) {
    var param_str = Array();
    for (let key in kwargs) {
        param_str.push(key + "=" + kwargs[key])
    }
    return "?" + param_str.join("&")
};

let get_index = function (arr, descend=true) {
    let ori_index = arr.map((d,i)=>i);
    let tmp_zip = d3.zip(ori_index, arr);
    tmp_zip = tmp_zip.sort((a,b)=>descend?b[1]-a[1]:a[1]-b[1]);
    return tmp_zip.map(d=>d[0]);
};

let get_detail_info = function (model_info) {
    let op_name = operations;
    if (model_info === null) {
        let pattern_keys = [];
        for (let name of op_name) {
            pattern_keys.push([name]);
        }
        for (let name1 of op_name) {
            for (let name2 of op_name) {
                pattern_keys.push([name1, name2]);
            }
        }
        pattern_keys.push(['skip']);
        return pattern_keys;
    }
    let adjacency, op;
    adjacency = model_info['A'];
    op = model_info['S'];
    return get_detail_info_new(adjacency, op);

};

let get_detail_info_new = function(adjacency, op){
    // return [[], [], [],   [], ..., [],   [skip:x]]

    let op_map = {};
    let op_name = operations;
    operations.forEach((name, i) => op_map[name] = i);
    // if (dataset_name === "nasbench201") {
    //     op_map = {
    //         'avg_pool3x3': 0,
    //         'conv1x1': 1,
    //         'conv3x3': 2,
    //     };
    //     op_name = ['avg_pool3x3', 'conv1x1', 'conv3x3'];
    // }
    // else if (dataset_name === "nasbench101") {
    //     op_map = {
    //         'maxpool3x3': 0,
    //         'conv1x1-bn-relu': 1,
    //         'conv3x3-bn-relu': 2,
    //     };
    //     op_name = ['maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu'];
    // }
    // else if (dataset_name === "hrtnet") {
    //     op_map = {
    //         'Conv.': 0,
    //         'HRP conv.': 1,
    //         'HRP conv. x2': 2,
    //         'Transformer': 3,
    //     };
    //     op_name = ['Conv.', 'HRP conv.', 'HRP conv. x2', 'Transformer',];
    // }

    let uni_gram = new Array(op_name.length).fill(0);
    let bi_gram = new Array(op_name.length ** 2).fill(0);
    for(let i = 1; i < op.length-1; i++){
        uni_gram[op_map[op[i]]]++;
        for(let j = i+1; j < op.length - 1; j++){
            bi_gram[op_map[op[i]] * op_name.length + op_map[op[j]]] += adjacency[i][j];
        }
    }

    // let idx_uni = get_index(uni_gram);
    let idx_uni = uni_gram.map((_, i) => i);
    let uni_list = [];
    for (let i in idx_uni) {
        uni_list.push([op_name[idx_uni[i]], uni_gram[idx_uni[i]]]);
    }

    let idx_bi = get_index(bi_gram);
    let bi_list = [];
    for (let i in idx_bi) {
        bi_list.push([op_name[Math.floor(idx_bi[i] / op_name.length)], op_name[idx_bi[i] % op_name.length], bi_gram[idx_bi[i]]]);
    }
    uni_list = uni_list.concat(bi_list);

    // if(dataset_name === 'hrtnet'){
    //     uni_list.push(['skip', 0]);
    // } else {
        uni_list.push(['skip', get_skips(adjacency)]);
    // }

    return uni_list;
};

let get_skips = function (adjacency) {
    // return number of skips
    if(adjacency.length===2){
        return adjacency[0][1];
    }
    let in_adjacency = adjacency.map(d => new Array(d.length).fill(0));
    for(let i=0; i<adjacency.length-1; i++){
        let child = [];
        for(let j=0; j<adjacency[i].length; j++){
            if(adjacency[i][j]>=1){
                child.push(j);
            }
        }
        let idx = 0;
        while(idx<child.length){
            let tmp = child[idx];
            idx += 1;
            let tmp_child = [];
            for(let j=0; j<adjacency[tmp].length; j++){
                if(adjacency[tmp][j]>=1){
                    tmp_child.push(j);
                }
            }
            for(let j in tmp_child){
                in_adjacency[i][tmp_child[j]] = 1;
                if(child.indexOf((tmp_child[j]))===-1){
                    child.push(tmp_child[j]);
                }
            }
        }
    }
    let skip_num = 0;
    for(let i in adjacency){
        for(let j in adjacency){
            if(adjacency[i][j]>0 && in_adjacency[i][j]>0){
                skip_num += adjacency[i][j];
            }
        }
    }
    return skip_num;
};

let get_detail_info_original = function(adjacency, op){
    // return count of [p, c1, c3, p-p, p-c1, p-c3, c1-p, c1-c1, c1-c3, c3-p, c3-c1, c3-c3]
    let op_map = null;
    let op_name = null;
    if (dataset_name === "nasbench201") {
        op_map = {
            'avg_pool3x3': 0,
            'conv1x1': 1,
            'conv3x3': 2,
        };
        op_name = ['avg_pool3x3', 'conv1x1', 'conv3x3'];
    }
    else if (dataset_name === "nasbench101" || dataset_name === "nasbench301") {
        // return "details";
        op_map = {
            'maxpool3x3': 0,
            'conv1x1-bn-relu': 1,
            'conv3x3-bn-relu': 2,
        };
        op_name = ['maxpool3x3', 'conv1x1-bn-relu', 'conv3x3-bn-relu'];
    }
    else if (dataset_name === "hrtnet") {
        op_map = {
            'Conv.': 0,
            'HRP conv.': 1,
            'HRP conv. x2': 2,
            'Transformer': 3,
        };
        op_name = ['Conv.', 'HRP conv.', 'HRP conv. x2', 'Transformer',];
    }


    let uni_gram = new Array(op_name.length).fill(0);
    let bi_gram = new Array(op_name.length ** 2).fill(0);

    for(let i = 1; i < op.length-1; i++){
        let pre = 0, pre_count = 0, nxt = 0, nxt_count = 0;
        for(let j in op){
            if(adjacency[j][i]>0){
                pre_count += 1;
                pre = j
            }
            if(adjacency[i][j]>0){
                nxt_count += 1;
                nxt = j
            }
        }
        if(pre_count===1 && nxt_count===1 && parseInt(pre)===0 && parseInt(nxt)===op.length-1){
            uni_gram[op_map[op[i]]]++;
        }
    }

    for(let i = 1; i < op.length-1; i++){
        for(let j = i+1; j < op.length - 1; j++){
            bi_gram[op_map[op[i]] * op_name.length + op_map[op[j]]] += adjacency[i][j];
        }
    }
    for(let i in adjacency){
        for(let j in adjacency){
            if(j<=i && adjacency[i][j]>0){
                console.log('adja wrong');
            }
        }
    }
    return uni_gram.concat(bi_gram);
};

function createPdf(elem_id="body") {
  let options = {
      useCSS: true,
  };
  let area = document.getElementById(elem_id);
  let w = Number(area.getAttribute("width"));
  let h = Number(area.getAttribute("height"));
  // let w = 1920, h = 1080;
  let doc = new PDFDocument({compress: false, size: [w, h]});
  console.log(w, h);
  SVGtoPDF(doc, area, 0, 0, options);
  let stream = doc.pipe(blobStream());
  stream.on('finish', function() {
    let blob = stream.toBlob('application/pdf');
    if (navigator.msSaveOrOpenBlob) {
      navigator.msSaveOrOpenBlob(blob, 'File.pdf');
    } else {
      document.getElementById('pdf-file').contentWindow.location.replace(URL.createObjectURL(blob));
    }
  });
  doc.end();
}
