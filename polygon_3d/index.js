import * as THREE from 'https://cdn.skypack.dev/pin/three@v0.135.0-pjGUcRG9Xt70OdXl97VF/mode=imports,min/optimized/three.js';

// Size of canvas. These get updated to fill the whole browser.
let width = 150;
let height = 150;
let scene = null;
let camera = null;
let renderer = null;

// Constants
const numAgents = 5;
const visualRange = 200;
const speedLimit = 0.08;
const vel_multiplier = 0.0004; // adjust velocity by this %
const inertia = 8

// Social distance parameter
const D = 150

// Visual options
const DRAW_NEAR = true;
const DRAW_TRAIL = true;

var agents = [];

function randFloat(min, max) {
  return Math.random() * (max - min) + max
}

function randVect(min, max) {
  return new THREE.Vector3(randFloat(min, max), randFloat(min, max), randFloat(min, max))
}

function initAgents() {
  for (var i = 0; i < numAgents; i += 1) {
    const geometry = new THREE.BoxGeometry( 1, 1, 1 );
    const material = new THREE.MeshBasicMaterial( {color: 0x00ff00} );
    const cube = new THREE.Mesh( geometry, material );
    agents[i] = {
      vel: randVect(0, 0.01),
      near: [],
      history: [],
      object: cube,
    };
    scene.add( cube );
    cube.position.set(randFloat(-10, 10), randFloat(-10, 10), randFloat(-10, -40))
  }
}

function distance(agent1, agent2) {
  return Math.sqrt(
    (agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2 + (agent1.z - agent2.z) ** 2,
  );
}

// Returns all agents in visual range.
function visible(agent) {
  return agents.filter(other => distance(agent, other) < visualRange)
}

// Called initially and whenever the window resizes to update the canvas
// size and width/height variables.
function sizeCanvas() {
  width = window.innerWidth;
  height = window.innerHeight;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera( 75, width / height, 0.1, 1000 );
  renderer = new THREE.WebGLRenderer();
  renderer.setSize( width, height );
  document.body.appendChild( renderer.domElement );
}

// Attempt to maintain distance of D from all other agents
function socialDistance(agent) {

  let moveX = 0;
  let moveY = 0;
  let moveZ = 0;

  for (let other of agents) {
    if (other !== agent) {
      const vel_base = D - distance(agent, other)
      moveX += vel_base * (agent.x - other.x);
      moveY += vel_base * (agent.y - other.y);
      moveZ += vel_base * (agent.z - other.z);
    }
  }
  agent.vel.x += moveX * vel_multiplier;
  agent.vel.y += moveY * vel_multiplier;
  agent.vel.z += moveZ * vel_multiplier;
  agent.vel.x /= inertia
  agent.vel.y /= inertia
  agent.vel.z /= inertia
}

function limitSpeed(agent) {

  const speed = Math.sqrt(agent.vel.x ** 2 + agent.vel.y ** 2 + agent.vel.z ** 2);
  if (speed > speedLimit) {
    agent.vel.x = (agent.vel.x / speed) * speedLimit;
    agent.vel.y = (agent.vel.y / speed) * speedLimit;
    agent.vel.z = (agent.vel.z / speed) * speedLimit;
  }
}

// Main animation loop
function animationLoop() {

  for (let agent of agents) {
    // Update the velocities according to each rule
    socialDistance(agent);
    limitSpeed(agent);

    // Update the position based on the current velocity
    const oldPos = agent.object.position
    agent.object.position.set(oldPos.x + 0.1, oldPos.y, oldPos.z)
  }

  // Schedule the next frame
  renderer.render( scene, camera );
  window.requestAnimationFrame(animationLoop);
}

window.onload = () => {
  // Make sure the canvas always fills the whole window
  // window.addEventListener("resize", sizeCanvas, false);
  sizeCanvas();

  // Randomly distribute the agents to start
  initAgents();
  renderer.render(scene, camera)
  // Schedule the main animation loop
  window.requestAnimationFrame(animationLoop);
};