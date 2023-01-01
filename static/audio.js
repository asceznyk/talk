const uploadBtn = document.getElementById("upload");
const checkpointSelect = document.getElementById("checkpoint");
const taskSelect = document.getElementById("task");
const audioInp = document.getElementById("audio");
const audioPlayer = document.getElementById("player");
const statusDiv = document.getElementById("status");
const transcriptDiv = document.getElementById("transcript");

console.log('welcome to talk!')

async function sendPOST(url, formData) {
	uploadBtn.disabled = true;
	let result = await fetch(url, {method:"POST", body:formData});
	result = await result.json();
	uploadBtn.removeAttribute('disabled');
	console.log(result)
	return result
}

async function selectCkpt() {
	statusDiv.innerHTML = `loading checkpoint..`

	let formData = new FormData();
	formData.append("checkpoint", this.value);
	let result = await sendPOST('/checkpoint/', formData);
	
	statusDiv.innerHTML = result.status	
}

async function transcribeAudio(task) {
	transcriptDiv.innerHTML = `transcribing...`

	let formData = new FormData();
	let inpAudio = audioInp.files[0]

	if (inpAudio.type.startsWith('audio')) {
		formData.append("task", task);
		formData.append("audio", inpAudio);
		audioPlayer.src = URL.createObjectURL(inpAudio);
		let result = await sendPOST('/', formData);
		transcriptDiv.innerHTML = result.text	
	} else {
		transcriptDiv.innerHTML = `incorrect file type: ${inpAudio.type}! expected audio file.`
	}
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

customSelect("selectopts");
document.addEventListener("click", closeAllSelect);
checkpointSelect.addEventListener("change", selectCkpt);
uploadBtn.addEventListener("click", () => {transcribeAudio(taskSelect.value)});



