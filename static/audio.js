const uploadBtn = document.getElementById("upload");
const taskSelect = document.getElementById("task");
const audioInp = document.getElementById("audio");
const audioPlayer = document.getElementById("player");
const transcriptDiv = document.getElementById("transcript");

console.log('welcome to talk!')

//let task = taskSelect.value;

async function transcribeAudio(task) {
	transcriptDiv.innerHTML = `transcribing...`

	let formData = new FormData();
	let inpAudio = audioInp.files[0]

	if (inpAudio.type.startsWith('audio')) {
		formData.append("task", task);
		formData.append("audio", inpAudio);
		audioPlayer.src = URL.createObjectURL(inpAudio);

		let result = await fetch('/', {method:"POST", body:formData});
		result = await result.json()
		transcriptDiv.innerHTML = result.text	
	} else {
		transcriptDiv.innerHTML = `incorrect file type: ${inpAudio.type}! expected audio file.`
	}
}

uploadBtn.addEventListener("click", () => {transcribeAudio(taskSelect.value)});



