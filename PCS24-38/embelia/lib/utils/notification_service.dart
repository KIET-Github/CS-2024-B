import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class NotificationServices {
  final FlutterLocalNotificationsPlugin _flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  final AndroidInitializationSettings _androidInitializationSettings =
      const AndroidInitializationSettings('background');

  Future initialiseNotification() async {
    InitializationSettings initializationSettings =
        InitializationSettings(android: _androidInitializationSettings);

    await _flutterLocalNotificationsPlugin.initialize(initializationSettings);
  }

  Future sendNotification(String title, String body) async {
    NotificationDetails notificationDetails = const NotificationDetails(
        android: AndroidNotificationDetails("channelId", "channelName",
            importance: Importance.max,
            priority: Priority.high,
            showWhen: false,
            icon: 'background'));
    await _flutterLocalNotificationsPlugin.show(
        0, title, body, notificationDetails);
  }

  Future scheduleNotification(String title, String body) async {
    NotificationDetails notificationDetails = const NotificationDetails(
        android: AndroidNotificationDetails("channelId", "channelName",
            importance: Importance.max,
            priority: Priority.high,
            showWhen: false,
            icon: 'background'));
    await _flutterLocalNotificationsPlugin.periodicallyShow(
        0, title, body, RepeatInterval.hourly, notificationDetails);
  }

  Future cancelNotification() async {
    await _flutterLocalNotificationsPlugin.cancelAll();
  }

}
