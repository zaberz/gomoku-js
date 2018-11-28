const Trainpipeline = require('./train');

let board_width = 3,
    board_height = 3,
    n_in_row = 3;
let best_model = `file://./current_model_${board_width}*${board_height}_${n_in_row}/model.json`;
// let best_model = null;
let train = new Trainpipeline({width: board_width, height: board_height, n_in_row: n_in_row}, best_model);
train.run();
