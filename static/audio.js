const uploadBtn = document.getElementById("upload");
const audioInp = document.getElementById("audio");
const audioPlayer = document.getElementById("player");
const transcriptDiv = document.getElementById("transcript");

async function uploadAudio(e) {
	let formData = new FormData();
	let inpAudio = audioInp.files[0]

	formData.append("reqtype", "upload");
	formData.append("audio", inpAudio);
	audioPlayer.src = URL.createObjectURL(inpAudio);

	let result = await fetch('/', {method:"POST", body:formData});
	result = await result.json()
	transcriptDiv.innerHTML = result.text
}

uploadBtn.addEventListener("click", uploadAudio);




