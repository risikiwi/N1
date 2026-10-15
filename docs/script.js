let model = null;

/* load weights */
async function loadModel() {
    const res = await fetch("weights.json");
    model = await res.json();
    document.getElementById("status").textContent = "Network loaded. Start drawing.";
}

/* activation func & forward pass*/
// sigmoid activation only, for now
function activate(name, vec) {
    return vec.map(v => 1 / (1+Math.exp(-v)));
}

// returns activations of every layer
function forward(input) {
    let x = input;
    const a = [input];
    for (let i=0; i < model.layers.length; i++) {
        let layer = model.layers[i];
        let z = [];
        for (let j=0; j < layer.W.length; j++) {
            let row = layer.W[j];
            let s = layer.b[j];
        
            for (let k = 0; k < row.length; k++) {
                s += row[k] * x[k];
            }
            z.push(s)
        }
        x = activate(model.activations[i], z);
        a.push(x);
    }
    return a;
}

function argmax(arr) {
    return arr.indexOf(Math.max(...arr));
}
// verify with values in tests.json
async function selfTest() {
    const tests = await (await fetch("tests.json")).json();
    for (let i = 0; i < tests.inputs.length; i++) {
        let x = tests.inputs[i];
        const out = forward(x).at(-1);
        console.log(`sample ${i}: JS says ${argmax(out)}, Python said ${tests.predictions[i]}`);
    }
}

loadModel().then(selfTest);

