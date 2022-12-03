var actors;
var stepsize;
var playfieldWidth;
var playfield;
var myPlayfield;
var food_count;
var timeStep = 160;
var roundTo = 4;
var agent;
var textFile = '';
var model;
var data = [];
var actions = [];
var rewards = [];
var values = [];
var features = [];

var positions = {
	pacman: {
		coordinates: [0, 0],
		lastValidPos: [0, 0],
		leftHome: true,
		eaten_pill: false,
		lastDirection: '0',
	},
	red: { coordinates: [0, 0], lastValidPos: [0, 0], leftHome: true },
	pink: { coordinates: [7, 34], lastValidPos: [0, 0], leftHome: false },
	blue: { coordinates: [7, 34], lastValidPos: [0, 0], leftHome: false },
	yellow: { coordinates: [7, 34], lastValidPos: [0, 0], leftHome: false },
};
var pills = [
	[24, 40],
	[24, 480],
	[64, 120],
	[120, 40],
	[120, 480],
];
const action_list = ['up', 'down', 'left', 'right'];

class Model {
	constructor(num_inputs, hidden_size, num_actions) {
		this.num_inputs = num_inputs;
		this.hidden_size = hidden_size;
		this.num_actions = num_actions;

		this.actor = this.build_actor();
		this.critic = this.build_critic();
	}

	//TODO:might add glorotinitializer

	build_critic() {
		const input = tf.input({ shape: [this.num_inputs] });

		//critic network
		const value_fn_hidden = tf.layers
			.dense({
				units: this.hidden_size,
				activation: 'relu',
				name: 'critic_hidden',
			})
			.apply(input);
		const value_fn = tf.layers
			.dense({ units: 1, activation: 'linear', name: 'critic_output' })
			.apply(value_fn_hidden);
		return tf.model({ inputs: input, outputs: value_fn });
	}

	build_actor() {
		const input = tf.input({ shape: [this.num_inputs] });

		//policy network
		const policy_hidden = tf.layers
			.dense({
				units: this.hidden_size,
				activation: 'relu',
				name: 'policy_hidden',
			})
			.apply(input);
		const logits = tf.layers
			.dense({
				units: this.num_actions,
				activation: 'softmax',
				name: 'policy_output',
			})
			.apply(policy_hidden);
		return tf.model({ inputs: input, outputs: logits });
	}

	//returns float
	forward_pass(inputs) {
		let as_tensor = tf.tensor(inputs, [1, inputs.length]);
		return [this.actor.predict(as_tensor), this.critic.predict(as_tensor)];
	}

	actionFromDistribution(logits) {
		//tf.squeeze(tf.multinomial(logits, 1), (axis = -1));
		return tf.multinomial(logits, 1);
	}

	action_value(observation) {
		let [logits, value] = this.forward_pass(observation);
		let action = this.actionFromDistribution(logits);
		//console.log('action: ' + action, 'value: ' + value);
		return [action, value];
	}
}

class RewardHandler {
	constructor(initialFeatures) {
		this.action;
		this.currentFeatures = initialFeatures;
		this.prevFeatures;
		this.cooldown = { eaten_by_ghost: 5000, ate_a_ghost: 2000, won: 5000 };
		this.gotEaten = 0;
		this.ateGhost = 0;
		this.timeWon = 0;
		this.lastAction;
		this.is_dead = false;
	}

	update(currentFeatures, action) {
		this.prevFeatures = this.currentFeatures;
		this.currentFeatures = currentFeatures;
		this.lastAction = this.action;
		this.action = action;
	}

	is_win() {
		// works
		if (
			g.dotsRemaining == 0 &&
			Date.now() - this.timeWon > this.cooldown['won']
		) {
			console.log('won');
			this.timeWon = Date.now();
			return 500;
		} else {
			if (g.dotsEaten < 20) {
				return 1;
			}
			if (g.dotsEaten < 40) {
				return 2;
			}
			if (g.dotsEaten < 80) {
				return 3;
			}
			if (g.dotsEaten < 100) {
				return 4;
			}
			if (g.dotsEaten > 100) {
				return 5;
			}
			return 0;
		}
	}

	is_lose() {
		//works,
		for (var ghost = 1; ghost < g.playerCount + 4; ghost++) {
			if (
				Date.now() - this.gotEaten > this.cooldown['eaten_by_ghost'] &&
				g.actors[ghost].tilePos[0] == g.actors[0].tilePos[0] &&
				g.actors[ghost].tilePos[1] == g.actors[0].tilePos[1]
			) {
				let modes = new Set([8, 16, 32, 64, 128]);
				if (!modes.has(g.actors[ghost].mode)) {
					console.log('lost');
					this.gotEaten = Date.now();
					return [-500, true];
				}
			}
		}
		return [0, false];
	}

	ate_ghost() {
		for (var ghost = 1; ghost < g.playerCount + 4; ghost++) {
			if (
				Date.now() - this.gotEaten > this.cooldown['ate_a_ghost'] &&
				g.actors[ghost].tilePos[0] == g.actors[0].tilePos[0] &&
				g.actors[ghost].tilePos[1] == g.actors[0].tilePos[1]
			) {
				if (g.actors[ghost].mode == 4) {
					console.log('ate a ghost');
					this.ateGhost = Date.now();
					return 20;
				}
			}
		}
		return 0;
	}

	ate_food() {
		//works
		if (this.prevFeatures[7] < this.currentFeatures[7]) {
			return 12;
		} else {
			return 0;
		}
	}

	ate_powerPill() {
		//works
		if (this.prevFeatures[8] < this.currentFeatures[8]) {
			console.log('ate a powerpill');
			return 5;
		} else {
			return 0;
		}
	}

	is_reverse() {
		//only triggers when reversing direction
		if (this.action != getOppositeDirection(this.lastAction)) {
			return 0;
		} else {
			return -8;
		}
	}

	get_reward() {
		let lost = this.is_lose();
		this.is_dead = lost[1];
		return (
			this.is_win() +
			lost[0] +
			this.ate_ghost() +
			this.ate_food() +
			this.ate_powerPill() +
			this.is_reverse()
		);
	}
}

function getOppositeDirection(direction) {
	if (direction == 'no') {
		return 'no';
	}
	let oppositeOfAction = ['down', 'up', 'right', 'left'];
	return oppositeOfAction[action_list.indexOf(direction)];
}

class StateRepresenter {
	constructor() {
		this.total_food = 275;
		this.total_scared_time = g.levels.frightTotalTime + 46;
		this.maximum_path_length = 56;
		this.eaten_food = 0;
		this.prev_direction = 'left';
		this.eaten_pills = 0;
	}

	reset() {
		this.prev_direction = 'left';
		this.eaten_pills = 0;
		this.eaten_food = 0;
	}

	getInitialFeatures() {
		return [0, 0, 0, 1, 0, 0, 0, 0, 0];
	}

	ateFood() {
		this.eaten_food = g.dotsEaten;
	}

	atePill() {
		console.log('pill eaten');
		this.time_eaten_powerPill = Date.now() - 1;
		this.eaten_pills = this.eaten_pills + 1;
	}

	//4.1 ->
	//1: everything eaten
	//:0 nothing eaten
	lvl_progress() {
		return g.dotsEaten / this.total_food;
	}

	//4.2
	//1: long time to eat ghost -> 0: ghosts not scared anymore
	eval_powerPill() {
		//console.log(Date.now() - this.time_eaten_powerPill, g.actors[1].mode , positions["red"].isScared)
		if (Date.now() - this.time_eaten_powerPill < this.total_scared_time * 10) {
			return (
				1 -
				(Date.now() - this.time_eaten_powerPill) / (this.total_scared_time * 10)
			);
		} else {
			return 0;
		}
	}

	//4.3
	//returs normalized distance to next food
	//1: food is close
	//0: food is far
	eval_food(distanceToFood) {
		return (
			(this.maximum_path_length - distanceToFood) / this.maximum_path_length
		);
	}

	//4.4
	//1: ghost is far
	//0: ghost is close
	eval_ghost(distanceToGhost) {
		return (
			1 -
			(this.maximum_path_length - distanceToGhost) / this.maximum_path_length
		);
	}

	//4.5
	//1: scared ghost is very close
	//0: no scared ghost available
	eval_scared_ghost(timeOfPowerPill, distanceToScaredGhost) {
		if (timeOfPowerPill > 0) {
			return (
				(this.maximum_path_length - distanceToScaredGhost) /
				this.maximum_path_length
			);
		} else {
			return 0;
		}
	}

	//4.6
	//1: next Powerpill is close
	//0: next Powerpill is far
	eval_next_powerPill(distanceToPowerpill) {
		return (
			(this.maximum_path_length - distanceToPowerpill) /
			this.maximum_path_length
		);
	}

	//4.7
	//1: still current direction
	//0: changed direction
	eval_direction(current_direction, prev_direction) {
		if (current_direction != getOppositeDirection(prev_direction)) {
			return 1; //1
		} else {
			return 0;
		}
	}

	//evaluates distance from myPos to food, powerpill, scaredghost and ghost
	getDistances(myPos, timeOfPowerPill, predict) {
		let distances = {};
		distances['food'] = -1;
		distances['powerpill'] = -1;
		distances['ghost'] = -1;
		if (timeOfPowerPill > 0) {
			distances['scaredghost'] = -1;
			for (let i = 1; i < 5; i++) {
				if (positions[translateActors(i)].inNormalMode) {
					distances['ghost'] = -1;
					return shortestPathBFStoAll(myPos, distances, predict);
				}
			}
		} else {
			distances['scaredghost'] = this.maximum_path_length; //dont need distance to scared ghosts, bc there are non
		}
		return shortestPathBFStoAll(myPos, distances, predict);
	}

	//returns the abstract state representation in form of a feature vector
	getFeaturesForState(currentState, curr_direction, prev_direction) {
		let features = [];
		features.push(this.lvl_progress());
		features.push(this.eval_powerPill());

		let distances = this.getDistances(currentState, features[1], false);
		features.push(this.eval_food(distances['food']));
		features.push(this.eval_ghost(distances['ghost']));
		features.push(
			this.eval_scared_ghost(features[1], distances['scaredghost'])
		);
		features.push(this.eval_next_powerPill(distances['powerpill']));
		features.push(this.eval_direction(curr_direction, prev_direction));
		features.push(g.dotsEaten); //used in ate_food()
		features.push(this.eaten_pills); // not used
		return features;
	}

	getSuccessorFeatures(currentState, curr_direction, prev_direction) {
		let currentFeatures = this.getFeaturesForState(
			currentState,
			curr_direction,
			prev_direction
		);
		let succ = {};
		let successorStates = getSuccessors(currentState); // object, succesorStates["up"] gives upper successor state
		for (var direction in successorStates) {
			let features = [];
			//1: progress
			if (successorStates[direction].hasDot != 'no') {
				// food in direction -> increase progress
				features.push(currentFeatures[0] + 1 / this.total_food);
			} else {
				features.push(currentFeatures[0]);
			}
			//2: time till powerpill is gone
			if (successorStates[direction].hasDot == 'pill') {
				//next state has pill
				features.push(1);
			} else if (currentFeatures[1] == 0) {
				//does not have pill and has not eaten pill
				features.push(0);
			} else {
				//does not have pill but has eaten one
				features.push(
					Math.max(
						currentFeatures[1] - timeStep / (this.total_scared_time * 10)
					),
					0
				); //not sure
			}
			let distances = this.getDistances(
				successorStates[direction],
				features[1],
				true
			);
			//3: distance to food
			features.push(this.eval_food(distances['food']));
			//4: distance to closest ghost
			features.push(this.eval_ghost(distances['ghost'])); //inaccurate bc does not respecct movement of ghost
			//5: distance to scared ghost
			features.push(
				this.eval_scared_ghost(features[1], distances['scaredghost'])
			); //same here
			//6: distance to next powerpill
			features.push(this.eval_next_powerPill(distances['powerpill']));
			//7: if direction is current direction
			features.push(this.eval_direction(direction, curr_direction));

			succ[direction] = features;
		}
		return succ;
	}
}

//at state
//get action (acoording to current weights) explore or exploit
//do action
//new state
//get reward for new state
//update weights
class Agent {
	constructor() {
		this.epsilon = 0.9;
		this.learning_rate = 0.1;
		this.gamma = 0.5;
		this.weight_bound = 100;
		this.stateRep = new StateRepresenter();
		this.currentStateFeatures = roundFeatures(
			this.stateRep.getInitialFeatures()
		);
		this.prevStateFeatures;
		this.rewardHandler = new RewardHandler(this.currentStateFeatures);
		this.weights = new Array(7).fill(0);
		this.currentState =
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			];
		this.prev_direction;
		this.curr_direction = getDirection(0);
		this.currentReward;
	}

	newLife() {
		this.prev_direction;
		this.curr_direction = getDirection(0);
		this.currentStateFeatures = roundFeatures(
			this.stateRep.getInitialFeatures()
		);
		this.prevStateFeatures;
		this.stateRep.reset();
		this.rewardHandler.update(this.currentStateFeatures, this.curr_direction);
	}
	//updates the currentState
	update() {
		this.prev_direction = this.curr_direction;
		this.curr_direction = getDirection(0);
		this.currentState =
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			];
	}

	//function that takes step and udates values
	takeStep() {
		let myPos =
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			];
		this.successorFeatures = this.stateRep.getSuccessorFeatures(
			myPos,
			this.curr_direction,
			this.prev_direction
		);
		this.successorFeatures = roundFeatures(this.successorFeatures);
		let action = this.getPolicy();

		applyAction(action);
		this.update();
		//console.log(action);
		stateUpdater(this.stateRep);
		this.prevStateFeatures = this.currentStateFeatures.slice();
		updatePositions(this.stateRep);
		this.currentStateFeatures = this.stateRep.getFeaturesForState(
			this.currentState,
			this.curr_direction,
			this.prev_direction
		);
		this.currentStateFeatures = roundFeatures(this.currentStateFeatures);

		this.rewardHandler.update(this.currentStateFeatures, action);

		this.updateWeights();
		data.push({
			features: this.currentStateFeatures,
			action: action,
			reward: this.currentReward,
		});

		return this.currentStateFeatures;
	}

	//return an action according to an epsilon greedy policy
	getPolicy() {
		let rng = Math.random();
		if (rng < this.epsilon) {
			return this.getMaxQ(this.successorFeatures, false);
		} else {
			//include only valid actions
			let validActions = [];
			for (var direction in this.successorFeatures) {
				validActions.push(direction);
			}
			return getRandomElement(validActions);
		}
	}

	//returns the maxQ value for all features in the successorFeatures array, flag value, if max value or best action should be returned
	getMaxQ(successorFeatures, value) {
		let max = -100000;
		let bestAction;
		let data = [];
		for (var direction in successorFeatures) {
			let sum = 0;
			for (let i = 0; i < this.weights.length; i++) {
				sum = sum + this.weights[i] * successorFeatures[direction][i];
			}
			data.push([sum, direction]);
			if (sum > max) {
				max = sum;
				bestAction = direction;
			}
		}
		if (value) {
			return max;
		} else {
			return bestAction;
		}
	}
	//updates weights according to following rule:
	//w(i) = w(i) + alpha*[reward + discount_factor*maxQ - prevQ]*feature(i)
	updateWeights() {
		this.currentReward = this.rewardHandler.get_reward();
		let oldQ = 0;
		for (let i = 0; i < this.weights.length; i++) {
			oldQ = oldQ + this.weights[i] * this.prevStateFeatures[i];
		}
		let error =
			this.currentReward +
			this.gamma * this.getMaxQ(this.successorFeatures, true) -
			oldQ;
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i] =
				this.weights[i] +
				this.learning_rate * error * this.prevStateFeatures[i];
		}

		//normalize weights, to avoid overfitting
		let sum = 0;
		for (let i = 0; i < this.weights.length; i++) {
			sum = sum + this.weights[i] * this.weights[i];
		}
		sum = Math.sqrt(sum);
		if (sum > this.weight_bound) {
			for (let i = 0; i < this.weights.length; i++) {
				this.weights[i] = (this.weight_bound * this.weights[i]) / sum;
			}
		}
	}
}

class neuralController {
	constructor() {
		this.gamma = 0.95;
		this.learning_rate = 0.005;
		this.value_c = 0.5;
		this.neuralModel = new Model(7, 50, 4);
		this.neuralModel.actor.compile({
			optimizer: tf.train.adam(this.learning_rate),
			loss: this.logits_loss,
		});
		this.neuralModel.critic.compile({
			optimizer: tf.train.adam(this.learning_rate),
			loss: 'meanSquaredError',
		});
	}

	logits_loss(actions_and_advantages, logits) {
		let advantages_pos = tf.max(actions_and_advantages, 1, true);
		let advantages_neg = tf.min(actions_and_advantages, 1, true);
		let advantages = tf.add(advantages_pos, advantages_neg);
		let one_hot_actions = tf.div(actions_and_advantages, advantages);
		let cross_entropy = tf.losses.softmaxCrossEntropy(one_hot_actions, logits);
		let weighted_cross_entropy = tf.mul(cross_entropy, advantages);
		weighted_cross_entropy.print();
		return weighted_cross_entropy;
	}

	return_advantages() {
		let returns = new Array(rewards.length);
		returns[rewards.length - 1] = rewards[rewards.length - 1];
		for (let i = rewards.length - 2; i >= 0; i--) {
			returns[i] = rewards[i] + this.gamma * returns[i + 1];
		}
		let advantages = returns.map((v, i) => v - values[i]);
		return [returns, advantages];
	}
}

class neuralAgent {
	constructor() {
		this.controller = new neuralController();
		this.epsilon = 0.9;
		this.stateRep = new StateRepresenter();
		this.currentStateFeatures = roundFeatures(
			this.stateRep.getInitialFeatures()
		);
		this.rewardHandler = new RewardHandler(this.currentStateFeatures);
		this.currentState =
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			];
		this.prev_direction;
		this.curr_direction = getDirection(0);
	}

	newLife() {
		this.prev_direction;
		this.curr_direction = getDirection(0);
		this.currentStateFeatures = roundFeatures(
			this.stateRep.getInitialFeatures()
		);
		this.stateRep.reset();
		this.rewardHandler.update(this.currentStateFeatures, this.curr_direction);
		this.train();
	}
	//updates the currentState
	update() {
		this.prev_direction = this.curr_direction;
		this.curr_direction = getDirection(0);
		this.currentState =
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			];
	}

	//function that takes step and udates values
	takeStep() {
		let myPos =
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			];
		this.currentStateFeatures = this.stateRep.getFeaturesForState(
			myPos,
			this.curr_direction,
			this.prev_direction
		);
		this.currentStateFeatures = roundFeatures(this.currentStateFeatures);
		const [action_tens, value_tens] = this.controller.neuralModel.action_value(
			this.currentStateFeatures.slice(0, 7)
		);
		let action = action_list[action_tens.dataSync()[0]];
		//console.log(action_tens.dataSync()[0]);

		applyAction(action);
		this.update();
		stateUpdater(this.stateRep);
		updatePositions(this.stateRep);
		this.rewardHandler.update(this.currentStateFeatures, action);
		this.remember_step(
			action,
			value_tens,
			this.rewardHandler.get_reward(),
			this.currentStateFeatures.slice(0, 7)
		);

		if (this.rewardHandler.is_dead) {
			this.currentStateFeatures = roundFeatures(
				this.stateRep.getInitialFeatures()
			);
		}
		return this.currentStateFeatures;
	}

	remember_step(action, value_tens, reward, feature) {
		actions.push(action_list.indexOf(action));
		values.push(value_tens.dataSync()[0]);
		rewards.push(reward);
		features.push(feature);
		data.push({ features: feature, action: action, reward: reward });
	}

	//function that trains network
	train() {
		let [returns, advantages] = this.controller.return_advantages();
		let one_hot_actions = tf.oneHot(actions, 4);
		one_hot_actions = tf.transpose(one_hot_actions);
		one_hot_actions = tf.mul(one_hot_actions, advantages);
		one_hot_actions = tf.transpose(one_hot_actions);
		let actor_loss = this.controller.neuralModel.actor.fit(
			tf.tensor(features),
			one_hot_actions,
			{
				batch_size: 32,
				epochs: 1,
				shuffle: true,
			}
		);
		let critic_loss = this.controller.neuralModel.critic.fit(
			tf.tensor(features),
			tf.tensor(returns, [returns.length, 1]),
			{
				batch_size: 32,
				epochs: 1,
				shuffle: true,
			}
		);
		console.log('actor_loss:' + actor_loss, 'critic_loss:' + critic_loss);
		actions = [];
		values = [];
		rewards = [];
		features = [];
	}
}

//function that rounds feature according to preference
function roundFeatures(features) {
	if (features instanceof Array) {
		return features.map(function (element) {
			return Number(element.toFixed(roundTo));
		});
	}
	let newFeatures = {};
	for (var dir in features) {
		newFeatures[dir] = features[dir].map(function (element) {
			return Number(element.toFixed(roundTo));
		});
	}
	return newFeatures;
}

//main function that runs the agent
async function myfunction() {
	/* 	var neuralController = new Model(7, 50, 4);
	tfvis.show.modelSummary({ name: 'Model Summary' }, neuralController.model);
	myPlayfield = buildPlayField();
	agent = new Agent();
	updatePositions(agent.stateRep);
	let currentfeatures;
	while (true) {
		currentfeatures = agent.takeStep();
		await checkForPause(currentfeatures);
		if (g.lives == -1) {
			console.log('died after eating:', agent.stateRep.eaten_food);
			startNew();
			await timeOut(2500);
			myPlayfield = buildPlayField();
			updatePositions(agent.stateRep);
			agent.newLife();
		}
	} */
	myPlayfield = buildPlayField();
	agent = new neuralAgent();
	/* tfvis.show.modelSummary(
		{ name: 'Actor Summary' },
		agent.controller.neuralModel.actor
	);
	tfvis.show.modelSummary(
		{ name: 'Critic Summary' },
		agent.controller.neuralModel.critic
	); */
	updatePositions(agent.stateRep);
	let currentfeatures;
	while (true) {
		currentfeatures = agent.takeStep();
		await checkForPause(currentfeatures, agent.rewardHandler.is_dead);
		if (g.lives == -1) {
			console.log('died after eating:', agent.stateRep.eaten_food);
			startNew();
			await timeOut(2500);
			myPlayfield = buildPlayField();
			updatePositions(agent.stateRep);
			agent.newLife();
		}
	}

	myPlayfield = buildPlayField();
	console.log(myPlayfield);
	var stateRep = new StateRepresenter();

	let currentFeatures = stateRep.getInitialFeatures();
	console.log(currentFeatures);
	let lastState = currentFeatures;
	var rewardHandler = new RewardHandler(currentFeatures);
	var prev_direction;
	var curr_direction = getDirection(0);
	updatePositions(stateRep);
	while (true) {
		prev_direction = curr_direction;
		curr_direction = getDirection(0);
		lastState = currentFeatures.slice();
		stateUpdater(stateRep);
		currentFeatures = stateRep.getFeaturesForState(
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			],
			curr_direction,
			prev_direction
		);
		console.log(currentFeatures[2]);
		let successorFeatures = stateRep.getSuccessorFeatures(
			myPlayfield[positions['pacman'].coordinates[0]][
				positions['pacman'].coordinates[1]
			],
			getDirection(0),
			positions['pacman'].lastDirection
		);
		//console.log(successorFeatures)
		rewardHandler.update(currentFeatures, '');
		console.log(
			rewardHandler.get_reward(),
			rewardHandler.is_win(),
			rewardHandler.is_lose(),
			rewardHandler.ate_ghost(),
			rewardHandler.ate_food(),
			rewardHandler.ate_powerPill(),
			rewardHandler.is_reverse()
		);
		await checkForPause(currentFeatures);
		updatePositions(stateRep);
		if (g.lives == -1) {
			startNew();
			await timeOut(2500);
			myPlayfield = buildPlayField();
			console.log('relive');
			updatePositions(stateRep);
			stateRep.reset();
		}
		/* let action = getBestAction(successorFeatures) 
        doAction(action) */
	}
}

function readWeights() {
	try {
		let file = document.querySelector('#weightFile').files[0];
		let reader = new FileReader();
		reader.readAsText(file, 'UTF-8');
		reader.onload = function () {
			let text_array = reader.result.split('#');
			let last_entry = text_array.pop().split(' ')[2];
			console.log(last_entry);
			agent.weights = last_entry.split(',').map(Number);
			console.log('successfully uploaded! Weights now: ' + agent.weights);
		};

		reader.onerror = function () {
			console.log(reader.error);
		};
	} catch {
		console.log('error reading weights');
	}
}

function saveWeights() {
	console.log(textFile);
	var blob = new Blob([textFile.toString()], {
		type: 'text/plain;charset=utf-8',
	});
	var url = URL.createObjectURL(blob);

	var a = document.createElement('a');
	a.download = 'weights.txt';
	a.href = url;
	a.click();
}

//function that handles the start of a new game
function startNew() {
	g.insertCoin();
	for (let i = 2; i < 5; i++) {
		positions[translateActors(i)].leftHome = false;
	}
	g.levels.frightTime = 540;
	g.levels.frightTotalTime = 729;
	pills = [
		[24, 40],
		[24, 480],
		[64, 120],
		[120, 40],
		[120, 480],
	];
	//updates weight file
	/* let text = 'ate: ' + agent.stateRep.eaten_food + ' ';
	for (let i = 0; i < agent.weights.length; i++) {
		text = text + agent.weights[i] + ',';
	}
	text = text.substring(0, text.length - 1);
	text = text + '#\n';
	if (textFile.split('#').length > 25) {
		textFile = '';
	}
	textFile = textFile.concat(text);
	console.log(agent.weights); */
}

//helper funtion that returns a promise -> used for timeouts
function timeOut(ms) {
	return new Promise((res) => setTimeout(() => res('p2'), ms));
}

//function that returns a timeout according to what happend in game
function checkForPause(currentFeatures, is_dead) {
	if (g.dotsRemaining == 0) {
		console.log('timer, win');
		return new Promise((res) => setTimeout(() => res('p2'), 5000));
	}
	if (is_dead) {
		//distance to ghost
		return new Promise((res) => setTimeout(() => res('p2'), 5200));
	}
	if (currentFeatures[4] == 1) {
		//distance to scared ghost
		console.log('timer, ate ghost');
		return new Promise((res) => setTimeout(() => res('p2'), 2000));
	}
	return new Promise((res) => setTimeout(() => res('p2'), timeStep));
}

//helper function that presses the correct button for the action taken
function applyAction(direction) {
	switch (direction) {
		case 'up':
			return g.keyPressed(38);
		case 'down':
			return g.keyPressed(40);
		case 'right':
			return g.keyPressed(39);
		case 'left':
			return g.keyPressed(37);
		case 'no':
			return false;
	}
}

//function that updates the myPlayfield-variable to be consistent with game state
function stateUpdater(stateRep) {
	if (stateRep.eaten_food < g.dotsEaten) {
		stateRep.ateFood();
		myPlayfield[positions['pacman'].coordinates[0]][
			positions['pacman'].coordinates[1]
		].hasDot = 'no';
	}
}

//simple BFS  algorithm that calculates the shortest path from the pacman to various destinations
function shortestPathBFStoAll(start, distances, predict) {
	//console.log("to find ", distances)
	//if start or goal is a Wall, invalid distance
	if (start.isWall) {
		console.log('start is wall', start);
		return -1;
	}
	let queue = []; //push and shift
	queue.push({ state: start, distance: 0 }); //queue stores open list of states and distances
	let visited = {};
	for (let i = 0; i < myPlayfield.length; i++) {
		for (let j = 0; j < myPlayfield[0].length; j++) {
			visited[myPlayfield[i][j].coordinates] = false;
		}
	}
	visited[start.coordinates] = true;
	let distance = 0;
	let result;
	while (queue.length != 0) {
		let next = queue.shift();
		result = foundEverthing(next, distances, predict);
		if (result[0]) {
			//termination criteria
			return result[1];
		}
		let succOfNext = getSuccessors(next.state); //generate successors
		for (const dir in succOfNext) {
			if (!visited[succOfNext[dir].coordinates]) {
				queue.push({ state: succOfNext[dir], distance: next.distance + 1 });
				visited[succOfNext[dir].coordinates] = true;
			}
		}
		distance = distance + 1;
	}
	for (const dist in result[1]) {
		if (result[1][dist] == -1) {
			result[1][dist] = 56;
		}
	}
	return distances;
}

function predictedState(index) {
	let direction = getDirection(index);
	let succ = getSuccessors(positions[translateActors(index)]);
	if (direction in succ) {
		return succ[direction].coordinates; //state in direction
	}
	if (Object.keys(succ).length == 2) {
		for (dir of Object.keys(succ)) {
			if (dir != getOppositeDirection(direction)) {
				return succ[dir].coordinates; //state in only moving direction
			}
		}
	}
	return positions[translateActors(index)].coordinates; //current state
}

function foundEverthing(next, distances, predict) {
	if (distances['food'] == -1 && next.state.hasDot == 'food') {
		distances['food'] = next.distance;
	}
	if (distances['ghost'] == -1) {
		for (let i = 1; i < 5; i++) {
			if (predict) {
				if (
					positions[translateActors(i)].leftHome &&
					positions[translateActors(i)].inNormalMode &&
					arrayEquals(next.state.coordinates, predictedState(i))
				) {
					distances['ghost'] = next.distance;
					break;
				}
			} else {
				if (
					positions[translateActors(i)].leftHome &&
					positions[translateActors(i)].inNormalMode &&
					arrayEquals(
						next.state.coordinates,
						positions[translateActors(i)].coordinates
					)
				) {
					distances['ghost'] = next.distance;
					break;
				}
			}
		}
	}
	if (distances['scaredghost'] == -1) {
		for (let i = 1; i < 5; i++) {
			if (!predict) {
				if (
					positions[translateActors(i)].leftHome &&
					positions[translateActors(i)].isScared &&
					arrayEquals(next.state.coordinates, predictedState(i))
				) {
					distances['scaredghost'] = next.distance;
					break;
				}
			} else {
				if (
					positions[translateActors(i)].leftHome &&
					positions[translateActors(i)].isScared &&
					arrayEquals(
						next.state.coordinates,
						positions[translateActors(i)].coordinates
					)
				) {
					distances['scaredghost'] = next.distance;
					break;
				}
			}
		}
	}
	if (distances['powerpill'] == -1 && next.state.hasDot == 'pill') {
		distances['powerpill'] = next.distance;
	}

	for (const dist in distances) {
		if (distances[dist] == -1) {
			return [false, distances];
		}
	}
	return [true, distances];
}

//state is element of myPlayfield
function getSuccessors(state) {
	let succ = {};
	if (state.isWall) {
		return succ;
	}
	for (let x = -1; x < 2; x = x + 2) {
		try {
			if (!myPlayfield[state.coordinates[0] + x][state.coordinates[1]].isWall) {
				let dir;
				if (x == -1) {
					dir = 'up';
				} else {
					dir = 'down';
				}
				succ[dir] = myPlayfield[state.coordinates[0] + x][state.coordinates[1]];
				//succ.push(myPlayfield[state.coordinates[0] + x][state.coordinates[1]])
			}
		} catch {
			console.log('state', state.coordinates, 'has no x', x);
			console.log(state);
		}
	}

	for (let y = -1; y < 2; y = y + 2) {
		try {
			let dir;
			if (y == -1) {
				dir = 'left';
			} else {
				dir = 'right';
			}
			if (arrayEquals(state.coordinates, [8, 0]) && y == -1) {
				succ[dir] = myPlayfield[8][57];
				//succ.push(myPlayfield[8][57])
				continue;
			} else if (arrayEquals(state.coordinates, [8, 57]) && y == +1) {
				succ[dir] = myPlayfield[8][0];
				//succ.push(myPlayfield[8][0])
				continue;
			} else if (
				!myPlayfield[state.coordinates[0]][state.coordinates[1] + y].isWall
			) {
				succ[dir] = myPlayfield[state.coordinates[0]][state.coordinates[1] + y];
				//succ.push(myPlayfield[state.coordinates[0]][state.coordinates[1] +y])
			}
		} catch {
			console.log('state', state.coordinates, 'has no y', y);
		}
	}
	return succ;
}

//0: no dir
//1: up
//2: downconso
//4: left
//8: right
function getDirection(index) {
	switch (g.actors[index].dir) {
		case 0:
			return 'no';
		case 1:
			return 'up';
		case 2:
			return 'down';
		case 4:
			return 'left';
		case 8:
			return 'right';
	}
}

//function that mantains the positions of the ghosts and pacman
function updatePositions(stateRep) {
	positions['pacman'].lastDirection = getDirection(0);
	let actors = g.actors;
	for (let i = 0; i < 5; i++) {
		//only update positions when left home for first time
		if (positions[translateActors(i)].leftHome) {
			try {
				positions[translateActors(i)].lastValidPos = [
					actors[i].lastGoodTilePos[0] / 8,
					(actors[i].lastGoodTilePos[1] - 32) / 8,
				];
			} catch {
				positions[translateActors(i)].lastValidPos = [64 / 8, (280 - 32) / 8];
			}

			let x, y;
			if (Math.round((actors[i].pos[1] - 32) / 8 < 0)) {
				x = 8;
				y = 57;
			} else if (Math.round((actors[i].pos[1] - 32) / 8 > 57)) {
				x = 8;
				y = 0;
			} else {
				x = Math.round(actors[i].pos[0] / 8);
				y = Math.round((actors[i].pos[1] - 32) / 8);
			}
			if (!myPlayfield[x][y].isWall) {
				positions[translateActors(i)].coordinates = [x, y];
			} else {
				positions[translateActors(i)].coordinates =
					positions[translateActors(i)].lastValidPos;
				myPlayfield = buildPlayField();
			}
			if (i > 0) {
				// ghost only
				positions[translateActors(i)].inNormalMode =
					g.actors[i].mode == 1 || g.actors[i].mode == 2;
				positions[translateActors(i)].isScared = g.actors[i].mode == 4;
			}
		} else {
			if (Math.floor((actors[i].pos[1] - 32) / 8) < 5) {
				positions[translateActors(i)].leftHome = true;
			}
		}
	}
	for (let i = 0; i < pills.length; i++) {
		if (pills[i] == null) {
			continue;
		}
		if (g.playfield[pills[i][0]][pills[i][1]].dot != 2) {
			stateRep.atePill();
			pills[i] = null;
		}
	}
}

//function that can be relaced by a dictionary
//translates between index and name of actor
function translateActors(i) {
	switch (i) {
		case 0:
			return 'pacman';
		case 1:
			return 'red';
		case 2:
			return 'pink';
		case 3:
			return 'blue';
		case 4:
			return 'yellow';
	}
}

//function that updates the myPlayfield-variable
function buildPlayField() {
	let currG = g['playfield'].slice(0);
	food_count = [0, 0];
	let tempPlayfield = [];
	for (let row = 0; row < g.playfieldHeight + 2; row++) {
		let thisRow = [];
		for (let col = 0; col < g.playfieldWidth - 7; col++) {
			let currState = currG[row * 8][32 + col * 8];
			let thisState = {};
			if (!currState['path']) {
				thisState['isWall'] = true;
			} else {
				thisState['isWall'] = false;
			}
			if (currState['dot'] == 1) {
				food_count[0] = food_count[0] + 1;
				thisState['hasDot'] = 'food';
			} else if (currState['dot'] == 2) {
				food_count[1] = food_count[1] + 1;
				thisState['hasDot'] = 'pill';
			} else {
				thisState['hasDot'] = 'no';
			}
			if (currState['intersection']) {
				thisState['isCrossing'] = true;
			} else {
				thisState['isCrossing'] = false;
			}
			thisState['coordinates'] = [row, col];
			thisRow.push(thisState);
		}
		tempPlayfield.push(thisRow);
	}
	return tempPlayfield;
}

//helper function that checks for equality of arrays
function arrayEquals(a, b) {
	return (
		Array.isArray(a) &&
		Array.isArray(b) &&
		a.length === b.length &&
		a.every((val, index) => val === b[index])
	);
}

//finds longest path from any state to any other state on the playfield
//takes some time to run (2min), could be way more efficient, but is not needed
//because we only need to run it once, since the mayximal path length does not change
//-> 56
function findMaximalPathLength() {
	let max = 0;
	for (let i = 1; i < myPlayfield.length - 1; i++) {
		// for every state
		for (let j = 0; j < myPlayfield[0].length; j++) {
			let start = myPlayfield[i][j];
			console.log(i, j, max);
			for (let k = 1; k < myPlayfield.length - 1; k++) {
				// to every state
				for (let l = 0; l < myPlayfield[0].length; l++) {
					let goal = myPlayfield[k][l];
					let pathLength = shortestPathBFS(start, 'state', goal);
					if (pathLength >= max) {
						max = pathLength;
						console.log(i, j, k, l);
					}
				}
			}
		}
	}
	return max;
}

//function that counts the total number of food
//269
function countTotalFood() {
	let count = 0;
	for (let i = 0; i < myPlayfield.length; i++) {
		for (let j = 0; j < myPlayfield[i].length; j++) {
			if (myPlayfield[i][j].hasDot == 'food') {
				count = count + 1;
			}
		}
	}
	return count;
}

//returns int i in [min, max)
function getRandomInt(min, max) {
	min = Math.ceil(min);
	max = Math.floor(max);
	return Math.floor(Math.random() * (max - min)) + min;
}

//helper function that returns random element in array
function getRandomElement(array) {
	return array[getRandomInt(0, array.length)];
}
