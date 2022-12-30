const uploadBtn = document.getElementById("upload");
const audioInp = document.getElementById("audio");
const audioPlayer = document.getElementById("player");
const transcriptDiv = document.getElementById("transcript");

async function transcribeAudio(contentElem) {
	transcriptDiv.innerHTML = `transcrbing...`

	let formData = new FormData();
	let inpAudio = audioInp.files[0]

	if (inpAudio.type.startsWith('audio')) {
		formData.append("reqtype", "upload");
		formData.append("audio", inpAudio);
		audioPlayer.src = URL.createObjectURL(inpAudio);

		let result = await fetch('/', {method:"POST", body:formData});
		result = await result.json()
		transcriptDiv.innerHTML = result.text	
	} else {
		transcriptDiv.innerHTML = `incorrect file type: ${inpAudio.type}! expected audio file.`
	}
}

uploadBtn.addEventListener("click", transcriptDiv);




