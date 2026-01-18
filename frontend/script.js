const video = document.getElementById("video");
const detectEmotionBtn = document.getElementById("detectEmotionBtn");
const emotionText = document.getElementById("emotionText");
const audioPlayer = document.getElementById("audioPlayer");
let detectInterval = null;


const emotionMusic = {
  happy: "https://www.bensound.com/bensound-music/bensound-sunny.mp3",
  sad: "https://www.bensound.com/bensound-music/bensound-slowmotion.mp3",
  angry: "https://www.bensound.com/bensound-music/bensound-energy.mp3",
  fear: "https://www.bensound.com/bensound-music/bensound-relaxing.mp3",
  neutral: "https://www.bensound.com/bensound-music/bensound-ukulele.mp3",
  disgust: "https://www.bensound.com/bensound-music/bensound-epic.mp3",
  surprise: "https://www.bensound.com/bensound-music/bensound-creativecommons.mp3",
};

detectEmotionBtn.addEventListener("click", async () => {
  // Capture current frame from video
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

  // Convert to base64
  const imageBase64 = canvas.toDataURL('image/jpeg');

  // Send to backend
  emotionText.textContent = "Detecting...";
  try {
    const response = await fetch('http://127.0.0.1:5000/detect_emotion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 })
    });
    const result = await response.json();
    if (result.emotion) {
      emotionText.textContent = result.emotion;
      const musicSrc = emotionMusic[result.emotion];
      if (musicSrc) {
        audioPlayer.src = musicSrc;
        audioPlayer.play();
      }
    } else {
      emotionText.textContent = 'Unknown';
      alert(result.error || 'Error detecting emotion');
    }
  } catch (err) {
    emotionText.textContent = 'Error';
    alert("Failed to connect to backend. Is the server running?");
  }
});

async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error("Error accessing webcam:", err);
    alert("Camera access denied or unavailable.");
  }
}

startWebcam();

const toggleBtn = document.createElement('button');
toggleBtn.textContent = "Start Live Detection";
toggleBtn.id = "toggleLiveBtn";
toggleBtn.style.marginLeft = "10px";
document.getElementById("controls")?.appendChild(toggleBtn);

toggleBtn.addEventListener("click", () => {
  if (detectInterval) {
    clearInterval(detectInterval);
    detectInterval = null;
    toggleBtn.textContent = "Start Live Detection";
  } else {
    // Trigger your existing click handler every 300ms
    detectInterval = setInterval(() => detectEmotionBtn.click(), 300);
    toggleBtn.textContent = "Stop Live Detection";
  }
});
