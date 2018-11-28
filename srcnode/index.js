const Trainpipeline = require('./train');

let best_model = 'indexeddb://best_policy_model';
best_model = null;
let train = new Trainpipeline({width: 3, height: 3, n_in_row: 3}, best_model);
train.run();
