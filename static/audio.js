const uploadBtn = document.getElementById("upload");
const audioInp = document.getElementById("audio");

async function uploadAudio(e) {
	let formData = new FormData(); 
	formData.append("reqtype", "upload");
	formData.append("audio", audioInp.files[0]);
	console.log(formData);
	await fetch('/', {method:"POST", body:formData});	
}

uploadBtn.addEventListener("click", uploadAudio);




