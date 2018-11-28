import {Board, Game} from './game';
import PolicyValueNet  from './policy_value_net_tensorflow';
import MCTSPlayer from './mcts_alphaZero';
import MCTS_Pure from './mcts_pure';
import _ from 'lodash';
import nj from 'numjs';
import tf from '@tensorflow/tfjs';

let policy_value_net = new PolicyValueNet(3, 3, 'indexeddb://best_policy_model_3*3_3');

let current_mcts_player = new MCTSPlayer(policy_value_net.policy_value_fn.bind(policy_value_net), 5, 500)

// let current_mcts_player = new MCTSPlayer(()=>{}, 5, 500);
let pure_mcts_player = new MCTS_Pure(5, 1000);

let win_cnt = {'-1': 0, '0': 0, '1': 0,'2': 0};

let board = new Board(3, 3, 3);
let game = new Game(board);
let n_games = 20;
for (let i of _.range(n_games)) {
    let winner = game.start_play(current_mcts_player, pure_mcts_player, i % 2, 0);
    // let winner = game.start_play(pure_mcts_player, pure_mcts_player, i % 2, 0);
    console.log(winner);
    win_cnt[winner.toString()] += 1;
}
console.log(win_cnt);
let win_ratio = (win_cnt['1'] + 0.5 * win_cnt['-1']) / n_games;

console.log(win_ratio);