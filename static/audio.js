const uploadBtn = document.getElementById("upload");
const audioInp = document.getElementById("audio");

async function uploadAudio(e) {
	let formData = new FormData(); 
	formData.append("reqtype", "upload");
	formData.append("audio", audioInp.files[0]);	
	let result = await fetch('/', {method:"POST", body:formData});
	result = await result.json()
	console.log(result)
}

uploadBtn.addEventListener("click", uploadAudio);




