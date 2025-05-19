package com.example.mqtt_app;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import org.eclipse.paho.client.mqttv3.IMqttDeliveryToken;
import org.eclipse.paho.client.mqttv3.MqttCallback;
import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.MqttMessage;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import android.widget.ArrayAdapter;
import android.widget.ListView;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    private MqttClient client;
    private MqttConnectOptions options;
    private Handler handler;
    private ScheduledExecutorService scheduler;


    private UDPClient udpClient = new UDPClient();
    private ImageView imageView;
    private boolean isStreaming = false;


    private String productKey = "k0lrjCZ8vGA";
    private String deviceName = "pi_app";
    private String deviceSecret = "dc07f50808aad9108865b9d1e900e864";

    private final String pub_topic = "/sys/k0lrjCZ8vGA/pi_app/thing/event/property/post";
    private final String sub_topic = "/sys/k0lrjCZ8vGA/pi_app/thing/service/property/set";

    private Double temperature = 0.0;

    private String daodi_value;
    private String zhangaiwu_value;
    private String question_value;







    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);




        ListView history_talk_list = findViewById(R.id.history_talk);
        ListView zhangaiwu_list = findViewById(R.id.zhangaiwu);

        TextView daodi_state_text = findViewById(R.id.daodi_state);

        // 创建第一个列表的字符串列表和适配器
        List<String> historyMessages = new ArrayList<>();
        ArrayAdapter<String> historyAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, historyMessages);
        history_talk_list.setAdapter(historyAdapter);

        // 创建第二个列表的字符串列表和适配器
        List<String> zhangaiwuMessages = new ArrayList<>();
        ArrayAdapter<String> zhangaiwuAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, zhangaiwuMessages);
        zhangaiwu_list.setAdapter(zhangaiwuAdapter);







        mqtt_init();
        start_reconnect();

        handler = new Handler() {
            @SuppressLint("SetTextI18n")
            public void handleMessage(Message msg) {
                super.handleMessage(msg);
                switch (msg.what) {
                    case 1: //开机校验更新回传
                        break;
                    case 2:  // 反馈回传
                        break;
                    case 3:  //MQTT 收到消息回传   UTF8Buffer msg=new UTF8Buffer(object.toString());
                        // 将收到的消息对象转换为字符串
                        String message = msg.obj.toString();
                        // 记录收到的消息到Debug日志中
                        Log.d("nicecode", "handleMessage: " + message);
                        try {
                            //Toast.makeText(MainActivity.this, "收到消息", Toast.LENGTH_SHORT).show();
                            // 将收到的字符串消息转换为JSONObject对象
                            JSONObject jsonObjectALL = new JSONObject(message);
                            // 从整个JSON对象中获取名为"items"的子对象
                            JSONObject items = jsonObjectALL.getJSONObject("items");
                            // 从"items"对象中分别获取名为"temp"和"humi"的子对象

// 倒地
                            try {
                                JSONObject obj_daodi = items.getJSONObject("daodi");
                                daodi_value = obj_daodi.getString("value");
                                daodi_state_text.setText("当前状态：" + daodi_value);
                            } catch (JSONException e) {
                                Log.e("nicecode", "倒地解析错误: " + e.getMessage());
                                //Toast.makeText(MainActivity.this, "倒地解析失败", Toast.LENGTH_SHORT).show();
                            }

// 历史对话
                            try {
                                JSONObject obj_question = items.getJSONObject("question");
                                question_value = obj_question.getString("value");
                                historyMessages.add(question_value);
                                historyAdapter.notifyDataSetChanged();
                            } catch (JSONException e) {
                                Log.e("nicecode", "历史对话解析错误: " + e.getMessage());
                                //Toast.makeText(MainActivity.this, "历史对话解析失败", Toast.LENGTH_SHORT).show();
                            }
// 障碍物识别
                            try {
                                JSONObject obj_zhangaiwu = items.getJSONObject("zhangaiwu");
                                zhangaiwu_value = obj_zhangaiwu.getString("value");
                                zhangaiwuMessages.add(zhangaiwu_value);
                                zhangaiwuAdapter.notifyDataSetChanged();
                            } catch (JSONException e) {
                                Log.e("nicecode", "障碍物识别解析错误: " + e.getMessage());
                                //Toast.makeText(MainActivity.this, "障碍物识别解析失败", Toast.LENGTH_SHORT).show();
                            }
                             //显示
                        } catch (JSONException e) {
                            // 如果解析JSON时出现异常，打印异常信息到Debug日志中
                            e.printStackTrace();
                            Toast.makeText(MainActivity.this, "解析失败", Toast.LENGTH_SHORT).show();
                            break;
                        }
                        break;
                    case 30:  //连接失败
                        Toast.makeText(MainActivity.this, "阿里云连接失败", Toast.LENGTH_SHORT).show();
                        //mqtt_sign.setImageResource(R.drawable.yun_close); // 更改图片资源
                        break;
                    case 31:   //连接成功
                        Toast.makeText(MainActivity.this, "阿里云连接成功", Toast.LENGTH_SHORT).show();
                        //mqtt_sign.setImageResource(R.drawable.yun_open); // 更改图片资源
                        try {
                            client.subscribe(sub_topic, 1);
                        } catch (MqttException e) {
                            e.printStackTrace();
                        }
                        break;
                    default:
                        break;
                }
            }
        };
    }


    private void mqtt_init() {
        try {

            String clientId = "a1MoTKOqkVK.test_device1";
            Map<String, String> params = new HashMap<String, String>(16);
            params.put("productKey", productKey);
            params.put("deviceName", deviceName);
            params.put("clientId", clientId);
            String timestamp = String.valueOf(System.currentTimeMillis());
            params.put("timestamp", timestamp);
            // cn-shanghai
            String host_url = "tcp://" + productKey + ".iot-as-mqtt.cn-shanghai.aliyuncs.com:1883";
            String client_id = clientId + "|securemode=2,signmethod=hmacsha1,timestamp=" + timestamp + "|";
            String user_name = deviceName + "&" + productKey;
            String password = com.example.mqtt_app.AliyunIoTSignUtil.sign(params, deviceSecret, "hmacsha1");

            //host为主机名，test为clientid即连接MQTT的客户端ID，一般以客户端唯一标识符表示，MemoryPersistence设置clientid的保存形式，默认为以内存保存
            System.out.println(">>>" + host_url);
            System.out.println(">>>" + client_id);

            //connectMqtt(targetServer, mqttclientId, mqttUsername, mqttPassword);

            client = new MqttClient(host_url, client_id, new MemoryPersistence());
            //MQTT的连接设置
            options = new MqttConnectOptions();
            //设置是否清空session,这里如果设置为false表示服务器会保留客户端的连接记录，这里设置为true表示每次连接到服务器都以新的身份连接
            options.setCleanSession(false);
            //设置连接的用户名
            options.setUserName(user_name);
            //设置连接的密码
            options.setPassword(password.toCharArray());
            // 设置超时时间 单位为秒
            options.setConnectionTimeout(10);
            // 设置会话心跳时间 单位为秒 服务器会每隔1.5*20秒的时间向客户端发送个消息判断客户端是否在线，但这个方法并没有重连的机制
            options.setKeepAliveInterval(60);
            //设置回调
            client.setCallback(new MqttCallback() {
                @Override
                public void connectionLost(Throwable cause) {
                    //连接丢失后，一般在这里面进行重连
                    System.out.println("connectionLost----------");
                }

                @Override
                public void deliveryComplete(IMqttDeliveryToken token) {
                    //publish后会执行到这里
                    System.out.println("deliveryComplete---------" + token.isComplete());
                }

                @Override
                public void messageArrived(String topicName, MqttMessage message)
                        throws Exception {
                    //subscribe后得到的消息会执行到这里面
                    System.out.println("messageArrived----------");
                    Message msg = new Message();
                    //封装message包
                    msg.what = 3;   //收到消息标志位
                    msg.obj = message.toString();
                    //发送messge到handler
                    handler.sendMessage(msg);    // hander 回传
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void mqtt_connect() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    if (!(client.isConnected()))  //如果还未连接
                    {
                        client.connect(options);
                        Message msg = new Message();
                        msg.what = 31;
                        // 没有用到obj字段
                        handler.sendMessage(msg);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    Message msg = new Message();
                    msg.what = 30;
                    // 没有用到obj字段
                    handler.sendMessage(msg);
                }
            }
        }).start();
    }

    private void start_reconnect() {
        scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                if (!client.isConnected()) {
                    mqtt_connect();
                }
            }
        }, 0 * 1000, 10 * 1000, TimeUnit.MILLISECONDS);
    }

    private void publish_message(String message) {
        if (client == null || !client.isConnected()) {
            return;
        }
        MqttMessage mqtt_message = new MqttMessage();
        mqtt_message.setPayload(message.getBytes());
        try {
            client.publish(pub_topic, mqtt_message);
        } catch (MqttException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        udpClient.stopReceiving();
    }
}