package com.example.mqtt_app;
import java.io.ByteArrayOutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class UDPClient {
    private DatagramSocket socket;
    private InetAddress serverAddress;
    private int serverPort;
    private boolean isReceiving = false;
    private Thread receiveThread;

    public void connect(String serverIp, int port) {
        try {
            socket = new DatagramSocket();
            serverAddress = InetAddress.getByName(serverIp);
            serverPort = port;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void sendData(String data) {
        if (socket == null || serverAddress == null || serverPort == 0) {
            System.out.println("未连接到服务器！");
            return;
        }

        new Thread(() -> {
            try {
                byte[] sendData = data.getBytes();
                DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length, serverAddress, serverPort);
                socket.send(sendPacket);
                System.out.println("发送数据: " + data);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }

    public void receiveImageStream(ImageStreamListener listener) {
        receiveThread = new Thread(() -> {
            try {
                byte[] buffer = new byte[1500];
                Map<Integer, ByteArrayOutputStream> frameBuffers = new HashMap<>();
                Map<Integer, Integer> framePacketCounts = new HashMap<>();

                isReceiving = true;

                while (isReceiving) {
                    DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                    socket.receive(packet);

                    byte[] receivedData = packet.getData();
                    int packetLength = packet.getLength();

                    ByteBuffer headerBuffer = ByteBuffer.wrap(receivedData, 0, 6);
                    int frameId = headerBuffer.getInt();
                    int packetIndex = headerBuffer.get();
                    int totalPackets = headerBuffer.get();

                    byte[] packetData = Arrays.copyOfRange(receivedData, 6, packetLength);

                    frameBuffers.putIfAbsent(frameId, new ByteArrayOutputStream());
                    framePacketCounts.putIfAbsent(frameId, totalPackets);

                    frameBuffers.get(frameId).write(packetData);

                    if (framePacketCounts.get(frameId) == packetIndex + 1) {
                        byte[] imageData = frameBuffers.get(frameId).toByteArray();
                        listener.onImageReceived(imageData);

                        frameBuffers.remove(frameId);
                        framePacketCounts.remove(frameId);
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        receiveThread.start();
    }

    public void stopReceiving() {
        isReceiving = false;
        if (receiveThread != null) {
            receiveThread.interrupt();
            try {
                receiveThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public interface ImageStreamListener {
        void onImageReceived(byte[] imageData);
    }
}
