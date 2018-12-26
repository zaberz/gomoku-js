const nj = require('numjs');

const tf = require('@tensorflow/tfjs');

const {zip} = require('./utils');

export default class PolicyValueNet {
    constructor(board_width, board_height, model_file = null) {
        this.board_width = board_width;
        this.board_height = board_height;

        this.optimizer = tf.train.adam();

        this.l2_const = 1e-4;
        this.create_policy_value_net();
        this._loss_train_op();

        // if (model_file) {
        //     this.restore_model(model_file);
        // }
    }

    create_policy_value_net() {
        let in_x,
            network,
            policy_net,
            value_net;

        in_x = network = tf.input({shape: [4, this.board_width, this.board_height]});

        // conv layers
        network = tf.layers.conv2d({
            filters: 32,
            kernelSize: [3, 3],
            padding: 'same',
            dataFormat: 'channelsFirst',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(network);

        network = tf.layers.conv2d({
            filters: 64,
            kernelSize: [3, 3],
            padding: 'same',
            dataFormat: 'channelsFirst',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(network);

        network = tf.layers.conv2d({
            filters: 128,
            kernelSize: [3, 3],
            padding: 'same',
            dataFormat: 'channelsFirst',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(network);

        // action policy layers

        policy_net = tf.layers.conv2d({
            filters: 4,
            kernelSize: [1, 1],
            dataFormat: 'channelsFirst',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(network);

        policy_net = tf.layers.flatten().apply(policy_net);
        this.policy_net = tf.layers.dense({
            units: this.board_width * this.board_height,
            activation: 'softmax',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(policy_net);

        value_net = tf.layers.conv2d({
            filters: 2,
            kernelSize: [1, 1],
            dataFormat: 'channelsFirst',
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(network);
        value_net = tf.layers.flatten().apply(value_net);
        value_net = tf.layers.dense({
            units: 64,
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(value_net);

        this.value_net = tf.layers.dense({
            units: 1,
            activation: 'tanh',
            kernelRegularizer: tf.regularizers.l2({l2: this.l2_const}),
        }).apply(value_net);

        this.model = tf.model({
            inputs: in_x,
            outputs: [this.policy_net, this.value_net],
        });

    }

    policy_value(state_input) {
        let res = tf.tidy(() => {

            let state_input_union = tf.tensor(nj.array(state_input).tolist());

            let [policy_net, value_net] = this.model.predict(state_input_union);

            let policy = policy_net.dataSync();
            let value = value_net.dataSync();
            policy = nj.array([...policy]).reshape(policy_net.shape).tolist();
            return [policy, value];
        });
        return res;
    }

    policy_value_fn(board) {
        let legal_position = board.availables;
        let current_state = board.current_state();
        let [act_probs, value] = this.policy_value(current_state.reshape([-1, 4, this.board_width, this.board_height]));
        act_probs = act_probs[0];
        act_probs = legal_position.map(item => {
            return act_probs[item];
        });
        act_probs = zip(legal_position, act_probs);
        return [act_probs, value[0]];
    }

    _loss_train_op() {
        let losses = ['categoricalCrossentropy', 'meanSquaredError'];
        this.model.compile({
            optimizer: this.optimizer,
            loss: losses,
        });

    }

    async train_step(state_input, mcts_probs, winner, learning_rate) {
        let state_input_union = tf.tensor(state_input.tolist());
        let mcts_probs_union = tf.tensor(mcts_probs);
        let winner_union = tf.tensor(winner);

        function self_entropy(probs) {
            return 1;
            // return -nj.mean(nj.sum(probs * nj.log(probs + 1e-10), 1));
        }

        let [loss0, entropy] = tf.tidy(() => {

            let losses = this.model.evaluate(state_input_union, [mcts_probs_union, winner_union],
                {
                    batchSize: state_input.length,
                    varbose: 0,
                });

            let [action_probs, _] = this.model.predict(state_input_union);

            let entropy = self_entropy(action_probs);

            // todo
            // K.set_value(self.model.optimizer.lr, learning_rate)

            this.optimizer.learningRate = learning_rate;

            let loss = losses[0].dataSync();

            return [loss, entropy];
        });

        await this.model.fit(state_input_union, [mcts_probs_union, winner_union], {
            batchSize: state_input.length,
            verbose: 0,
        });

        let resLoss = loss0;

        state_input_union.dispose();
        mcts_probs_union.dispose();
        winner_union.dispose();

        return [resLoss, entropy];
    }

    async save_model(model_file) {
        await this.model.save(model_file);
    }

    async restore_model(model_file) {
        await tf.loadModel(model_file).then(model => {
            this.model = model;
            this._loss_train_op();
        }).catch(e => {
            console.log(`best model not found, auto create`);
            console.log(e);
        });
    }
};