const {Board, Game} = require('./game');
const PolicyValueNet = require('./policy_value_net_tensorflow');
const MCTSPlayer = require('./mcts_alphaZero');
const MCTS_Pure = require('./mcts_pure');
const _ = require('lodash');
const nj = require('numjs');
const {zip} = require('./utils');
const tf = require('@tensorflow/tfjs');

require('@tensorflow/tfjs-node');
const Trainpipeline = require('./train');

let pure_mcts_playout_num = 500;
//
// let policy_value_net = new PolicyValueNet(3, 3);
// let policy_value_fn = function (board) {
//     let action_probs = new Array(board.availables.length).fill(1 / board.availables.length);
//     return [zip(board.availables, action_probs), 0];
// };
// let current_mcts_player = new MCTSPlayer(policy_value_fn, 5, 5000);

let width = 6,
    height = 6,
    n_in_row = 4;
let file = `file://./best_model_${width}*${height}_${n_in_row}/model.json`;
// file = null;
let policy_value_net = new PolicyValueNet(width, width, file);
policy_value_net.restore_model(file).then(()=>{

let current_mcts_player = new MCTSPlayer(policy_value_net.policy_value_fn.bind(policy_value_net), 5, 400);

//
let pure_mcts_player = new MCTS_Pure(5, pure_mcts_playout_num);
// let pure_mcts_player2 = new MCTS_Pure(5, pure_mcts_playout_num);
//
let win_cnt = {'-1': 0, '0': 0, '1': 0, '2': 0};

let board = new Board(width, height, n_in_row);
let game = new Game(board);
let n_games = 10;

for (let i of _.range(n_games)) {
    let winner = game.start_play(current_mcts_player, pure_mcts_player, i % 2,);

    console.log(winner);
    win_cnt[winner.toString()] += 1;
}
console.log(win_cnt);
let win_ratio = (win_cnt['1'] + 0.5 * win_cnt['-1']) / n_games;

console.log(win_ratio);

})
//
// let best_model = null;
// let train = new Trainpipeline({width: 3, height: 3, n_in_row: 3}, best_model);
// train.policy_evaluate();
//