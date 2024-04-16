import 'package:flutter/cupertino.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LocalStorage extends ChangeNotifier {
  static SharedPreferences? prefs;

  static Future init() async {
    prefs = await SharedPreferences.getInstance();
    return prefs;
  }

  static Future<void> setString(String key, String value) async {
    await prefs?.setString(key, value);
  }

  static Future<void> setBool(String key, bool value) async {
    await prefs?.setBool(key, value);
  }

  static Future<void> setInt(String key, int value) async {
    await prefs?.setInt(key, value);
  }

  static Future<void> setDouble(String key, double value) async {
    await prefs?.setDouble(key, value);
  }

  static Future<void> setStringList(String key, List<String> value) async {
    await prefs?.setStringList(key, value);
  }

  static String? getString(String key) {
    return prefs?.getString(key);
  }

  static bool? getBool(String key) {
    return prefs?.getBool(key);
  }

  static int? getInt(String key) {
    return prefs?.getInt(key);
  }

  static double? getDouble(String key) {
    return prefs?.getDouble(key);
  }

  static List<String>? getStringList(String key) {
    return prefs?.getStringList(key);
  }

  static Future<void> remove(String key) async {
    await prefs?.remove(key);
  }

  static Future<void> clear() async {
    await prefs?.clear();
  }

  static Future<void> reload() async {
    await prefs?.reload();
  }
}
