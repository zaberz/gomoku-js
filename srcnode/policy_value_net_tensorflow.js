const nj = require('numjs');
require('@tensorflow/tfjs-node');

const tf = require('@tensorflow/tfjs');

const {zip} = require('./utils');

class logSoftmax extends tf.layers.Layer {
    constructor(config) {
        super(config);
    }

    call(inputs) {
        return tf.log(tf.softmax(inputs));
    }
}

logSoftmax.className = 'logSoftmax';

tf.serialization.registerClass(logSoftmax);

module.exports = class PolicyValueNet {
    constructor(board_width, board_height, model_file = null) {
        this.board_width = board_width;
        this.board_height = board_height;

        // todo add transpose
        // this.input_state = tf.input({shape: [4, board_width, board_height]});
        // self.input_state = tf.transpose(self.input_states, [0, 2, 3, 1])
        this.input_state = tf.input({shape: [board_width, board_height, 4]});

        this.conv1 = tf.layers.conv2d({
            filters: 32,
            kernelSize: [3, 3],
            padding: 'same',
            dataFormat: 'channelsLast',
            activation: 'relu',
        }).apply(this.input_state);

        this.conv2 = tf.layers.conv2d({
            filters: 64,
            kernelSize: [3, 3],
            padding: 'same',
            dataFormat: 'channelsLast',
            activation: 'relu',
        }).apply(this.conv1);

        this.conv3 = tf.layers.conv2d({
            filters: 128,
            kernelSize: [3, 3],
            padding: 'same',
            dataFormat: 'channelsLast',
            activation: 'relu',
        }).apply(this.conv2);

        this.action_conv = tf.layers.conv2d({
            filters: 4,
            kernelSize: [1, 1],
            padding: 'same',
            dataFormat: 'channelsLast',
            activation: 'relu',
        }).apply(this.conv3);

        this.action_conv_flat = tf.layers.flatten().apply(this.action_conv);

        // 3-2 Full connected layer, the output is the log probability of moves
        // on each slot on the board

        this.action_fc = tf.layers.dense({
            units: board_width * board_height,
            activation: 'logSoftmax',
        }).apply(this.action_conv_flat);

        this.evaluation_conv = tf.layers.conv2d({
            filters: 2,
            kernelSize: [1, 1],
            padding: 'same',
            dataFormat: 'channelsLast',
            activation: 'relu',
        }).apply(this.conv3);
        this.evaluation_conv_flat = tf.layers.flatten().apply(this.evaluation_conv);

        this.evaluation_fc1 = tf.layers.dense({units: 64, activation: 'relu'}).apply(this.evaluation_conv_flat);
        // output the score of evaluation on current state
        this.evaluation_fc2 = tf.layers.dense({units: 1, activation: 'tanh'}).apply(this.evaluation_fc1);

        this.model = tf.model({inputs: this.input_state, outputs: [this.action_fc, this.evaluation_fc2]});

        // this.labels = tf.input({shape: [1]});
        //
        // this.value_loss = (pred, label) => {
        //     return pred.sub(label).square().mean()
        // };
        //
        // this.mcts_probs = tf.input({shape: [board_height * board_width]});
        // this.policy_loss = () => tf.neg(tf.mean(tf.sum(tf.mul(this.mcts_probs, this.action_fc), 1)));
        let l2_penalty_beta = 1e-4;
        let l2_penalty = tf.regularizers.l2({l2: l2_penalty_beta});

        this.optimizer = tf.train.adam(this.learning_rate);
        if (model_file) {
            this.restore_model(model_file);
        }
    }

    policy_value(state_batch) {
        // input: a batch of states
        // output: a batch of action probabilities and state values
        let res = tf.tidy(() => {
            state_batch = nj.reshape(state_batch, [-1, 4, this.board_width, this.board_height]);
            state_batch = nj.transpose(state_batch, [0, 2, 3, 1]);
            let model = this.model;
            // let [log_act_probs, value] = model.predict(state_batch);
            let q = tf.tensor(state_batch.tolist());

            let [log_act_probs, value] = model.predict(q);

            // todo
            let act_probs = tf.exp(log_act_probs);
            value = value.dataSync()[0];
            return [act_probs, value];
        });
        return res;

    }

    policy_value_fn(board) {
        // input: board
        // output: a list of (action, probability) tuples for each available
        // action and the score of the board state
        let legal_position = board.availables;

        let current_state = board.current_state();
        let [act_probs, value] = this.policy_value(current_state);

        let a = act_probs.dataSync();
        a = legal_position.map(item => {
            return a[item];
        });
        act_probs = zip(legal_position, a);
        return [act_probs, value];

    };

    async train_step(state_batch, mcts_probs, winner_batch, lr) {

        this.optimizer = tf.train.adam(lr);
        let loss = function (pred, label) {
            return tf.neg(tf.mean(tf.sum(pred.mul(label), 1)));
        };

        this.model.compile({
            optimizer: this.optimizer,
            // categoricalCrossentropy
            // loss
            loss: [loss, 'meanSquaredError'],
        });

        // perform a training step
        state_batch = nj.transpose(state_batch, [0, 2, 3, 1]);
        state_batch = tf.tensor(state_batch.tolist());

        let mp = [];
        mcts_probs.map(item => {
            mp = mp.concat(item.selection.data);
        });
        mcts_probs = tf.tensor(mp, [mcts_probs.length, this.board_width * this.board_height]);

        // winner_batch = nj.reshape(winner_batch, (-1, 1));
        winner_batch = tf.tensor(winner_batch, [winner_batch.length, 1]);

        let res = await this.model.fit(state_batch, [mcts_probs, winner_batch], {}).catch(e => {
            console.error(e);
        });

        state_batch.dispose();
        mcts_probs.dispose();
        winner_batch.dispose();

        // let loss;
        // this.optimizer.minimize(() => {
        //     let lossT = this.loss(this.f(state_batch), [mcts_probs, winner_batch]);
        //     loss = lossT.dataSync()[0]
        //     return lossT;
        // });
        // todo
        return [res.history.loss[0], 1];
        // console.log(res);
    }

    async save_model(model_file) {
        await this.model.save(model_file);
    }

    restore_model(model_file) {
        tf.loadModel(model_file).then(model => {
            this.model = model;
        }).catch(e => {
            console.log(`best model not found, auto create`);
            console.log(e);
        });
    }
};