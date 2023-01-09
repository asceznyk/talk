const checkpointSelect = document.getElementById("checkpoint");
const taskSelect = document.getElementById("task");
const audioTag = document.getElementById("player");
const statusDiv = document.getElementById("status");
const transcriptDiv = document.getElementById("transcript");

const audioPlayer = document.querySelector(".audio-player");

let allChunks = [];
let allTexts = [];
let stopped = 0;

let guid = window.navigator.userAgent.replace(/\D+/g, '');

console.log('welcome to talk!')

async function sendPOST(url, formData) {
	checkpointSelect.classList.add("disabled");
	let result = await fetch(url, {method:"POST", body:formData});
	result = await result.json();
	audioPlayer.classList.remove("disabled");
	console.log(result)
	return result
}

async function selectCkpt(e) {
	statusDiv.innerHTML = `loading checkpoint..`
	let fd = new FormData();
	fd.append("checkpoint", e.currentTarget.value);
	let result = await sendPOST('/checkpoint/', fd);	
	statusDiv.innerHTML = result.status	
}

function customSelect(className) {
	let x, i, j, l, ll, selElmnt, a, b, c;
	x = document.getElementsByClassName(className);
	l = x.length;
	for (i = 0; i < l; i++) {
		selElmnt = x[i].getElementsByTagName("select")[0];
		ll = selElmnt.length;
		a = document.createElement("DIV");
		a.setAttribute("class", "select-selected");
		a.innerHTML = selElmnt.options[selElmnt.selectedIndex].innerHTML;
		x[i].appendChild(a);
		b = document.createElement("DIV");
		b.setAttribute("class", "select-items select-hide");
		for (j = 0; j < ll; j++) {
			c = document.createElement("DIV");
			c.innerHTML = selElmnt.options[j].innerHTML;
			c.addEventListener("click", function(e) {
					let y, i, k, s, h, sl, yl;
					s = this.parentNode.parentNode.getElementsByTagName("select")[0];
					sl = s.length;
					h = this.parentNode.previousSibling;
					for (i = 0; i < sl; i++) {
						if (s.options[i].innerHTML == this.innerHTML) {
							s.selectedIndex = i;
							h.innerHTML = this.innerHTML;
							y = this.parentNode.getElementsByClassName("same-as-selected");
							yl = y.length;
							for (k = 0; k < yl; k++) {
								y[k].removeAttribute("class");
							}
							this.setAttribute("class", "same-as-selected");
							let evt = document.createEvent("HTMLEvents");
							evt.initEvent("change", false, true);
							s.dispatchEvent(evt);
							break;
						}
					}
					h.click();
			});
			b.appendChild(c);
		}
		x[i].appendChild(b);
		a.addEventListener("click", function(e) {
			e.stopPropagation();
			closeAllSelect(this);
			this.nextSibling.classList.toggle("select-hide");
			this.classList.toggle("select-arrow-active");
		});
	}
}

function closeAllSelect(elmnt) {
  let x, y, i, xl, yl, arrNo = [];
  x = document.getElementsByClassName("select-items");
  y = document.getElementsByClassName("select-selected");
  xl = x.length;
  yl = y.length;
  for (i = 0; i < yl; i++) {
    if (elmnt == y[i]) {
      arrNo.push(i)
    } else {
      y[i].classList.remove("select-arrow-active");
    }
  }
  for (i = 0; i < xl; i++) {
    if (arrNo.indexOf(i)) {
      x[i].classList.add("select-hide");
    }
  }
}

function pauseAudio(audio, btn) {	
	btn.classList.remove("pause");
	btn.classList.add("play");
	audio.pause();
}

function pauseAudio(audio, btn) {	
	btn.classList.remove("pause");
	btn.classList.add("play");
	audio.pause();
}

function customAudioPlayer(audio) {
	const playBtn = audioPlayer.querySelector(".controls .toggle-play");
	const progressBar = audioPlayer.querySelector(".progress");
	const volumeBtn = audioPlayer.querySelector(".volume-button");
	const volumeEl = audioPlayer.querySelector(".volume-container .volume"); 

	playBtn.addEventListener(
		"click", () => {
			if (!audioPlayer.classList.contains('disabled')) {
				if (audio.paused) {
					playBtn.classList.remove("play");
					playBtn.classList.add("pause");
					audio.play();
				} else {
					pauseAudio(audio, playBtn);
				}	
			}
		},
		false
	);

	volumeBtn.addEventListener("click", () => {
		if (!audioPlayer.classList.contains('disabled')) {
			audio.muted = !audio.muted;
			if (audio.muted) {
				volumeEl.classList.remove("fa-volume-up");
				volumeEl.classList.add("fa-volume-off");
			} else {
				volumeEl.classList.add("fa-volume-up");
				volumeEl.classList.remove("fa-volume-off");
			}
		}
	});

	setInterval(() => {
		progressBar.style.width = audio.currentTime / audio.duration * 100 + "%";		
	}, 500);

	audio.onended = (e) => {
		audio.src = URL.createObjectURL(new Blob(allChunks));
		pauseAudio(audio, playBtn);
	}
}

function liveAudioSpeechRecognition(audio) {
	let sidx = 0;
	let startBtn = document.getElementById("start");
	let stopBtn = document.getElementById("stop");

	if (navigator.mediaDevices) {
		navigator.mediaDevices.getUserMedia({audio: true})
		.then((stream) => {
			const mediaRecorder = new MediaRecorder(stream, {
				mimeType: 'audio/webm; codecs=opus'
			})

			startBtn.onclick = () => {
				console.log('start recording');
				allChunks = [];
				allTexts = [];
				audio.src = "";
				transcriptDiv.innerHTML = `<span>annotating..</span>`;
				stopped = 0;
				mediaRecorder.start();
				startBtn.style.background = "red";
				startBtn.style.color = "white";
				audioPlayer.classList.add("disabled");
			}

			stopBtn.onclick = (e) => {
				console.log('stop recording');
				stopped = 1;
				mediaRecorder.stop();
				startBtn.style.background = "";
				startBtn.style.color = "black";	
				audio.src = URL.createObjectURL(new Blob(allChunks))
				audioPlayer.classList.remove("disabled");
			}

			setInterval(function() { 
				if(mediaRecorder.state == "recording") {
					mediaRecorder.stop();
					sidx++;
				}	
			}, 2000);

			mediaRecorder.ondataavailable = async (e) => {
				allChunks.push(e.data);
				if (!stopped) {
					let fd = new FormData();
					fd.append("audio", new Blob([e.data]), `${guid}_${sidx}.webm`);
					fd.append("task", taskSelect.value);

					console.log('resuming media and sending audio request..');

					mediaRecorder.start();

					let result = await sendPOST("/", fd);
					let text = result.text;
					if(!text.includes('err_msg')) {
						allTexts.push(text);
						transcriptDiv.innerHTML = `${allTexts.join(' ')}` 
					}
				} 	
			}
		})
		.catch((err) => {
			console.error(`The following error occurred: ${err}`);
		})
	}
}

customSelect("selectopts");
customAudioPlayer(audioTag);
liveAudioSpeechRecognition(audioTag);

document.addEventListener("click", closeAllSelect);

checkpointSelect.addEventListener("change", (e) => { 
	if(!e.currentTarget.classList.contains('disabled')) {
		selectCkpt(e); 
	}
});





