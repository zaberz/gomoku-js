const {Board, Game} = require('./game');
// const PolicyValueNet = require('./policy_value_net_tensorflow');
const PolicyValueNet = require('./policy_value_net_keras');
const MCTSPlayer = require('./mcts_alphaZero');
const MCTS_Pure = require('./mcts_pure');
const _ = require('lodash');
const nj = require('numjs');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const fs = require('fs');

module.exports = class Trainpipeline {
    constructor(config = {}) {
        let width = this.board_width = config.width;
        let height = this.board_height = config.height;
        let n_in_row = this.n_in_row = config.n_in_row;

        this.board = new Board(width, height, n_in_row);
        this.game = new Game(this.board);

        this.game_batch_num = 1500;
        this.batch_size = 512;
        // this.buffer_size = 2

        // this.batch_size = 2;
        this.check_freq = 20;
        this.epochs = 5;

        this.learn_rate = 2e-3;
        this.lr_multiplier = 1.0;
        this.temp = 1.0;
        this.n_playout = 400;
        this.c_puct = 5;
        this.buffer_size = 10000;
        this.data_buffer = [];
        this.play_batch_size = 1;
        this.kl_targ = 0.02;
        this.best_win_ratio = 0.0;
        this.pure_mcts_playout_num = 1000;
        // this.init_model = init_model;
    }

    async run(init_model) {
        let width = this.board_width;
        let height = this.board_height;
        if (init_model) {
            this.policy_value_net = new PolicyValueNet(width, height, init_model);
            await this.policy_value_net.restore_model(init_model);
        } else {
            this.policy_value_net = new PolicyValueNet(width, height);
        }
        this.mcts_player = new MCTSPlayer(this.policy_value_net.policy_value_fn.bind(this.policy_value_net), this.c_puct, this.n_playout);

        for (let i of _.range(this.game_batch_num)) {
            await this.collect_selfplay_data(this.play_batch_size);
            let memory_used = tf.memory();
            console.log(`batch i:${i},episode_len:${this.episode_len};num_tensors:${memory_used.numTensors};num_bytes: ${memory_used.numBytes}`);
            if (this.data_buffer.length > this.batch_size) {
                this.has_update = true;
                let [loss, entropy] = await this.policy_update();
            }

            if ((i + 1) % this.check_freq === 0 && this.has_update) {
                console.log(`current self-pley batch: ${i + 1}`);
                let win_ratio = this.policy_evaluate();
                console.log(win_ratio);
                console.log('SAVING!!!');
                await this.policy_value_net.save_model(`file://./current_model_${this.board_width}*${this.board_height}_${this.n_in_row}`);
                console.log('SAVED');
                if (win_ratio > this.best_win_ratio) {
                    console.log(`New best policy !`);
                    this.best_win_ratio = win_ratio;
                    // localStorage.setItem('best_win_ratio', win_ratio);
                    console.log('SAVING!!!');
                    await this.policy_value_net.save_model(`file://./best_model_${this.board_width}*${this.board_height}_${this.n_in_row}`);
                    console.log('SAVED');
                    if (this.best_win_ratio === 1 && this.pure_mcts_playout_num < 5000) {
                        this.pure_mcts_playout_num += 1000;
                        this.best_win_ratio = 0;
                    }
                }
            }
        }
    }

    async collect_selfplay_data(n_games = 1) {
        for (let i of _.range(n_games)) {
            let [winner, play_data] = await this.game.start_self_play(this.mcts_player, this.temp, 0);
            this.episode_len = play_data.length;

            // 获取相同情况棋盘数据
            play_data = this.get_equi_data(play_data);

            let length = play_data.length;
            if (this.data_buffer.length + length > this.buffer_size) {
                this.data_buffer = this.data_buffer.slice(this.data_buffer.length + length - 1000);
            }

            this.data_buffer.push(...play_data);
        }
    }

    get_equi_data(play_data) {
        let extend_data = [];

        for (let [state, mcts_probs, winner] of play_data) {
            for (let i of [1, 2, 3, 4]) {
                let equi_state = nj.stack([
                    nj.rot90(state.slice([0, 1]).reshape([this.board_width, this.board_height]), i),
                    nj.rot90(state.slice([1, 2]).reshape([this.board_width, this.board_height]), i),
                    nj.rot90(state.slice([2, 3]).reshape([this.board_width, this.board_height]), i),
                    nj.rot90(state.slice([3, 4]).reshape([this.board_width, this.board_height]), i),
                ]);
                let equi_mcts_prob = nj.rot90(nj.flip(mcts_probs.reshape(this.board_height, this.board_width), 0), i);
                extend_data.push([equi_state,
                    nj.flip(equi_mcts_prob, 0).flatten(),
                    winner]);

                let equi_state2 = nj.flip(equi_state, 2);
                let equi_mcts_prob2 = nj.flip(equi_mcts_prob, 1);
                extend_data.push([
                    equi_state2,
                    nj.flip(equi_mcts_prob2, 0).flatten(),
                    winner,
                ]);
            }
        }
        return extend_data;
    }

    async policy_update() {
        // update the policy-value net
        let mini_batch = _.sampleSize(this.data_buffer, this.batch_size);
        // this.data_buffer = [];
        let state_batch = [];
        let mcts_probs_batch = [];
        let winner_batch = [];
        for (let data of mini_batch) {
            state_batch = state_batch.concat(data[0].selection.data);
            mcts_probs_batch.push(data[1].tolist());
            winner_batch.push(data[2]);
        }

        state_batch = nj.array(state_batch).reshape(-1, 4, this.board_width, this.board_height);
        let [old_probs, old_v] = this.policy_value_net.policy_value(state_batch);
        old_probs = tf.tensor(old_probs);

        let loss;
        let entropy;
        let kl;
        let new_v;
        for (let i of _.range(this.epochs)) {

            [loss, entropy] = await this.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch,
                this.learn_rate * this.lr_multiplier,
                this.batch_size,
                // this.epochs,
            );

            let [new_probs, v] = this.policy_value_net.policy_value(state_batch);
            new_v = v;
            new_probs = tf.tensor(new_probs);
            kl = tf.tidy(() => {
                let kl = tf.mean(tf.sum(old_probs.mul(tf.log(old_probs.add(1e-10)).sub(tf.log(new_probs.add(1e-10)))), 1));
                return kl.dataSync()[0];
            });
            new_probs.dispose();
            if (kl > this.kl_targ * 4) {
                break;
            }

        }

        if (kl > this.kl_targ * 2 && this.lr_multiplier > 0.1) {
            this.lr_multiplier /= 1.5;
        } else if (kl < this.kl_targ / 2 && this.lr_multiplier < 10) {
            this.lr_multiplier *= 1.5;
        }

        let explained_var_old = (1 - Math.pow(nj.std(nj.array(winner_batch).subtract(nj.array([...old_v]).flatten())) / (nj.std(nj.array(winner_batch)) || 1), 2));
        let explained_var_new = (1 - Math.pow(nj.std(nj.array(winner_batch).subtract(nj.array([...new_v]).flatten())) / (nj.std(nj.array(winner_batch)) || 1), 2));
        console.log(`kl:${kl};lr_multiplier:${this.lr_multiplier};loss:${loss};entropy:${entropy};old_v:${explained_var_old};new_v:${explained_var_new};`);
        old_probs.dispose();

        return [loss, entropy];
    }

    policy_evaluate(n_games = 10) {
        let current_mcts_player = new MCTSPlayer(this.policy_value_net.policy_value_fn.bind(this.policy_value_net), this.c_puct, this.n_playout);
        let pure_mcts_player = new MCTS_Pure(5, this.pure_mcts_playout_num);

        let win_cnt = {'-1': 0, '0': 0, '1': 0, '2': 0};

        for (let i of _.range(n_games)) {
            console.log(`评价模型第${i}局`);
            let winner = this.game.start_play(current_mcts_player, pure_mcts_player, i % 2, 0);
            win_cnt[winner.toString()] += 1;
        }
        let win_ratio = (win_cnt['1'] + 0.5 * win_cnt['-1']) / n_games;
        console.log(win_ratio);
        return win_ratio;
    }
};