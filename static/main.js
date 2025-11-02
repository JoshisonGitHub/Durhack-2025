

// peer connections
var config = { sdpSemantics: 'unified-plan' };
var pc =  new RTCPeerConnection(config);

// data channel
var dc = null, dcInterval = null;

function createPeerConnection() {
	var config = { sdpSemantics: 'unified-plan' };

	pc = new RTCPeerConnection(config);

	// connect audio / video
	pc.addEventListener('track', (evt) => {
		document.getElementById('cam').srcObject = evt.streams[0];
	});

	return pc;
}


function negotiate() {
	return pc.createOffer().then((offer) => {
		return pc.setLocalDescription(offer);
	}).then(() => {
		// wait for ICE gathering to complete
		return new Promise((resolve) => {
			if (pc.iceGatheringState === 'complete') {
				resolve();
			} else {
				function checkState() {
					if (pc.iceGatheringState === 'complete') {
						pc.removeEventListener('icegatheringstatechange', checkState);
						resolve();
					}
				}
				pc.addEventListener('icegatheringstatechange', checkState);
			}
		});
	}).then(() => {
		var offer = pc.localDescription;
		console.log(offer);
		
		return fetch('/offer', {
			body: JSON.stringify({
				sdp: offer.sdp,
				type: offer.type,
				// video_transform: document.getElementById('video-transform').value
			}),
			headers: {
				'Content-Type': 'application/json'
			},
			method: 'POST'
		});
	}).then((response) => {
		return response.json();
	}).catch((e) => {
		alert(e);
	});
}



	pc = createPeerConnection();
	var time_start = null;
	
	const current_stamp = () => {
		if (time_start === null) {
			time_start = new Date().getTime();
			return 0;
		} else {
			return new Date().getTime() - time_start;
		}
	};
	facing = 'environment';
	// Build media constraints.
	const constraints = {
		audio: false,
		video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
	};
	
	
	
	const videoConstraints = constraints.video;


	constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;


	// Acquire media and start negotiation.
	if (constraints.video) {
		navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
			console.log(pc)
			stream.getTracks().forEach((track) => {
				
				pc.addTrack(track, stream);
			});
			return negotiate();
		}, (err) => {
			alert('Could not acquire media: ' + err);
		});
	} else {
		negotiate();
	}


function stop() {
	document.getElementById('stop').style.display = 'none';
		
	// close data channel
	if (dc) {
		dc.close();
	}

	// close transceivers
	if (pc.getTransceivers) {
		pc.getTransceivers().forEach((transceiver) => {
			if (transceiver.stop) {
				transceiver.stop();
			}
		});
	}
	
	// close local audio / video
	pc.getSenders().forEach((sender) => {
		sender.track.stop();
	});

	// close peer connection
	setTimeout(() => {
		pc.close();
	}, 500);
}