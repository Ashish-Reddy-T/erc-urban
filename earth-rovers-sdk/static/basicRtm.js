$(document).ready(function () {
  const APP_ID = $("#appid").val();
  const USER_ID = $("#uid").val();
  const TOKEN = $("#rtm_token").val();
  const channelName = $("#channel").val(); // Replace with your desired channel name
  const botUid = $("#bot_uid").val();

  // Create an instance of the Agora RTM client
  const rtmClient = AgoraRTM.createInstance(APP_ID);

  // Create an instance of an RTM channel
  const rtmChannel = rtmClient.createChannel(channelName);

  // Event listener for connection state changes.
  // The RTM session carries BOTH telemetry (MessageFromPeer -> window.rtm_data)
  // and control (sendMessageToPeer). If it drops, /data silently serves the
  // last message forever and /control goes nowhere, while video keeps playing.
  // So: expose the state, and log back in automatically after a drop.
  window.rtm_state = "INIT";
  window.rtm_reason = "";
  window.rtm_dead = false;
  let relogging = false;
  // NB: Agora RTM 1.x emits "ConnectionStateChanged"; the SDK originally
  // listened for "ConnectionStateChange", which never fires. Listen to both.
  const onRtmState = (newState, reason) => {
    console.log(
      "on connection state changed to " + newState + " reason: " + reason
    );
    window.rtm_state = newState;
    window.rtm_reason = reason;
    window.rtm_state_t = Date.now() / 1000;
    if (newState === "ABORTED" || newState === "DISCONNECTED") {
      window.rtm_dead = true;
      if (!relogging) {
        relogging = true;
        setTimeout(() => {
          rtmClient.logout().catch(() => {}).finally(() => {
            joinRTMChannel(USER_ID);
            relogging = false;
          });
        }, 2000);
      }
    }
    if (newState === "CONNECTED") {
      window.rtm_dead = false;
    }
  };
  rtmClient.on("ConnectionStateChanged", onRtmState);
  rtmClient.on("ConnectionStateChange", onRtmState);
  rtmClient.on("TokenExpired", () => {
    window.rtm_state = "TOKEN_EXPIRED";
    window.rtm_dead = true;
  });

  // Event listener for receiving a channel message
  rtmChannel.on("ChannelMessage", ({ text }, senderId) => {
    console.log("AgoraRTM msg from user " + senderId + " received: \n" + text);
  });

  rtmClient.on("MessageFromPeer", function (message, peerId) {
    const controls = JSON.parse(new TextDecoder().decode(message.rawMessage));
    // Only the bot we command may populate rtm_data. 2026-09-22: two rovers
    // (signatures e538dd66.. and 473ca229..) were publishing into this
    // session and /data alternated between them (battery 29 / 80), mixing
    // their wheel and IMU streams.
    window.rtm_peers = window.rtm_peers || {};
    window.rtm_peers[String(peerId)] = (Date.now() / 1000);
    if (String(peerId) !== String(botUid)) {
      window.rtm_other = controls;
      return;
    }
    const event = new CustomEvent("message-from-peer", { detail: controls });
    window.rtm_data = controls;
    window.rtm_data.timestamp = (Date.now() / 1000).toFixed(6);
    window.rtm_data.peer = String(peerId);
    const formattedMessage = formatMessage(controls);
    $("#messages").html(formattedMessage);
    document.dispatchEvent(event);

    if (controls.latitude && controls.longitude) {
      const latitude = parseFloat(controls.latitude);
      const longitude = parseFloat(controls.longitude);
      console.log("updating marker");
      window.updateMarker(latitude, longitude);
    }

    console.log(
      "AgoraRTM peer msg from user " + peerId + " received: \n",
      controls
    );
  });

  function formatMessage(jsonData) {
    let formattedMessage =
      '<div class="card"><div class="card-body"><h5 class="card-title">Message from Peer</h5><ul class="list-group">';
    for (const [key, value] of Object.entries(jsonData)) {
      formattedMessage += `<li class="list-group-item"><strong>${key}:</strong> ${value}</li>`;
    }
    formattedMessage += "</ul></div></div>";
    return formattedMessage;
  }

  // Function to join the RTM channel
  function joinRTMChannel(uid) {
    rtmClient
      .login({ token: TOKEN, uid: String(uid) })
      .then(() => {
        console.log("AgoraRTM client login success");
        // Join a channel
        rtmChannel
          .join()
          .then(() => {
            console.log("RTM Channel join success");
            // You can now send messages or set up more event listeners
          })
          .catch((error) => {
            console.log("Failed to join channel for error: " + error);
          });
      })
      .catch((err) => {
        console.log("AgoraRTM client login failure", err);
      });
  }

  // Function to send message
  function sendMessage(json) {
    const message = JSON.stringify(json);
    console.warn("sending message to bot", botUid);
    console.warn("message", message);
    rtmClient
      .sendMessageToPeer(
        {
          text: message,
        },
        botUid
      )
      .then(() => {
        console.warn("Message sent successfully:", message);
      })
      .catch((err) => {
        console.warn("Error sending message:", err);
      });
  }

  // Make the function globally accessible
  window.sendMessage = sendMessage;

  // Call joinRTMChannel with the user ID to start the process
  joinRTMChannel(USER_ID);
});

