// Size of canvas. These get updated to fill the whole browser.
let width = 150;
let height = 150;

// Constants
const numAgents = 6;
const visualRange = 200;
const speedLimit = 8;
const vel_multiplier = 0.004; // adjust velocity by this %
const inertia = 8

// Social distance parameter
const D = 150

// Visual options
const DRAW_NEAR = true;
const DRAW_TRAIL = true;

var agents = [];

function initAgents() {
  for (var i = 0; i < numAgents; i += 1) {
    agents[i] = {
      x: Math.random() * width,
      y: Math.random() * height,
      dx: Math.random() * 10 - 5,
      dy: Math.random() * 10 - 5,
      near: [],
      history: [],
    };
  }
}

function distance(agent1, agent2) {
  return Math.sqrt(
    (agent1.x - agent2.x) ** 2 + (agent1.y - agent2.y) ** 2,
  );
}

// Returns all agents in visual range.
function visible(agent) {
  return agents.filter(other => distance(agent, other) < visualRange)
}

// Called initially and whenever the window resizes to update the canvas
// size and width/height variables.
function sizeCanvas() {
  const canvas = document.getElementById("polygon_2d");
  width = window.innerWidth;
  height = window.innerHeight;
  canvas.width = width;
  canvas.height = height;
}

// Attempt to maintain distance of D from all other agents
function socialDistance(agent) {

  let moveX = 0;
  let moveY = 0;

  for (let other of agents) {
    if (other !== agent) {
      vel_base = D - distance(agent, other)
      moveX += vel_base * (agent.x - other.x);
      moveY += vel_base * (agent.y - other.y);
    }
  }
  agent.dx += moveX * vel_multiplier;
  agent.dy += moveY * vel_multiplier;
  agent.dx /= inertia
  agent.dy /= inertia
}

function limitSpeed(agent) {

  const speed = Math.sqrt(agent.dx * agent.dx + agent.dy * agent.dy);
  if (speed > speedLimit) {
    agent.dx = (agent.dx / speed) * speedLimit;
    agent.dy = (agent.dy / speed) * speedLimit;
  }
}


function drawAgent(ctx, agent) {
  const angle = Math.atan2(agent.dy, agent.dx);
  ctx.translate(agent.x, agent.y);
  ctx.rotate(angle);
  ctx.translate(-agent.x, -agent.y);
  ctx.fillStyle = "#558cf4";
  ctx.beginPath();
  ctx.moveTo(agent.x, agent.y);
  ctx.arc(agent.x, agent.y, 5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  if (DRAW_NEAR) {
    ctx.strokeStyle = "#558cf466";
    ctx.beginPath();
    for (const point of agent.near) {
      ctx.moveTo(agent.x, agent.y);
      ctx.lineTo(point[0], point[1]);
    }
    ctx.stroke();
  }

  if (DRAW_TRAIL) {
    ctx.strokeStyle = "rgba(64,11,20,0.4)";
    ctx.beginPath();
    ctx.moveTo(agent.history[0][0], agent.history[0][1]);
    for (const point of agent.history) {
      ctx.lineTo(point[0], point[1]);
    }
    ctx.stroke();
  }
}

// Main animation loop
function animationLoop() {

  for (let agent of agents) {
    // Update the velocities according to each rule
    socialDistance(agent);
    limitSpeed(agent);

    // Update the position based on the current velocity
    agent.x += agent.dx;
    agent.y += agent.dy;
    agent.near = []
    for (let other of visible(agent)) {
      agent.near.push([other.x, other.y])
    }

    agent.history.push([agent.x, agent.y])
    agent.history = agent.history.slice(-50);
  }

  // Clear the canvas and redraw all the agents in their current positions
  const ctx = document.getElementById("polygon_2d").getContext("2d");
  ctx.clearRect(0, 0, width, height);
  for (let agent of agents) {
    drawAgent(ctx, agent);
  }

  // Schedule the next frame
  window.requestAnimationFrame(animationLoop);
}

window.onload = () => {
  // Make sure the canvas always fills the whole window
  window.addEventListener("resize", sizeCanvas, false);
  sizeCanvas();

  // Randomly distribute the agents to start
  initAgents();

  // Schedule the main animation loop
  window.requestAnimationFrame(animationLoop);
};