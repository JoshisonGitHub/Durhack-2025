#include <WiFiS3.h>

const char* ssid = "Pixel_5953";  // or your hotspot SSID

IPAddress local_ip(10,188,91,150);   // Arduino static IP
IPAddress gateway(10,188,91,73);     // Default gateway
IPAddress subnet(255,255,255,0);     // Subnet mask

WiFiServer server(80);  // HTTP server on port 80
const int RELAY_PIN = 3;

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  WiFi.config(local_ip, gateway, subnet);
  WiFi.begin(ssid);

  Serial.println("Connecting...");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }

  Serial.println();
  Serial.println("Connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("HTTP server started on port 80");
}

void loop() {
  WiFiClient client = server.available();
  if (!client) return;

  String request = client.readStringUntil('\r');
  client.flush();

  if (request.indexOf("/shoot") != -1) {
    Serial.println("Shooting Nerf!");
    digitalWrite(RELAY_PIN, HIGH);
    delay(400);
    digitalWrite(RELAY_PIN, LOW);
    Serial.println("Done.");
  }

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/plain");
  client.println("Connection: close");
  client.println();
  client.println("OK");
  delay(1);
}