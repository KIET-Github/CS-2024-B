import 'dart:convert';

import 'package:bloc/bloc.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:embelia/authentication/user_auth.dart';
import 'package:embelia/geminiAI/secrets.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';

part 'chat_bot_state.dart';

class ChatBotCubit extends Cubit<ChatBotState> {
  final SpeechToText _speechToText = SpeechToText();
  bool _speechEnabled = false;
  Map<String, dynamic> userData = {};
  String lastWords = '';
  ChatBotCubit() : super(ChatBotInitial());

  Future<String?> main(String apiKey, lastWords) async {
    emit(ChatBotLoading());
    try {
      final model = GenerativeModel(model: 'gemini-pro', apiKey: apiKey);
      final prompt = lastWords +
          " " +
          "(Please answer only if the question is related to physical and mental health or similar topics.)" +
          " " +
          "(This is the UserData: $userData, if you want to use it.)";
      final content = [Content.text(prompt)];
      print("Prompt: $prompt");
      await model.generateContent(content).then((value) {
        emit(ChatBotLoaded(value.text ?? ""));
        return value.text;
      });
    } catch (e) {
      emit(ChatBotError(e.toString()));
    }
    return null;
  }

  Future<Map<String, dynamic>> getRecommendationTasks(
      BuildContext context) async {
    try {
      final model = GenerativeModel(model: 'gemini-pro', apiKey: apiKey);
      final prompt = '''
          Generate three task recommendations for improving physical and
          mental health based on the following individual's data: $userData. Each task 
          should be labeled with a taskName and a task length in integer hours. and 
          value should be returned in this format : 
          "[{"taskName": "Example 1", "taskLength": 3}, {"taskName": "Example 2", "taskLength": 2}, {"taskName": "Exmaple 3", "taskLength": 1}]". 
          DO NOT WRITE OR EXPLAIN ANYTHING ELSE
          ''';
      final content = [Content.text(prompt)];
      Map<String, dynamic> task = {};
      await model.generateContent(content).then((value) async {
        debugPrint(value.text.toString());
        // Parsing the JSON string and adding key-value pairs to the map
        List<dynamic> jsonList = jsonDecode(value.text!.toString());

        // Iterate using index
        for (int i = 0; i < jsonList.length; i++) {
          task.addAll({
            "taskName$i": jsonList[i]['taskName'].toString(),
            "taskLength$i": jsonList[i]['taskLength']
          });
        }
      });
      return task;
    } catch (e) {
      debugPrint(e.toString());
    }
    return {};
  }

  bool isListening() {
    emit(ChatBotListening());
    return _speechToText.isListening;
  }

  /// This has to happen only once per app
  void initSpeech() async {
    _speechEnabled = await _speechToText.initialize();
  }

  /// Each time to start a speech recognition session
  void startListening() async {
    await _speechToText.listen(onResult: _onSpeechResult);
  }

  /// Manually stop the active speech recognition session
  /// Note that there are also timeouts that each platform enforces
  /// and the SpeechToText plugin supports setting timeouts on the
  /// listen method.
  void stopListening() async {
    await _speechToText.stop();
  }

  /// This is the callback that the SpeechToText plugin calls when
  /// the platform returns recognized words.
  void _onSpeechResult(SpeechRecognitionResult result) {
    if (result.finalResult) {
      lastWords = result.recognizedWords;
      main(apiKey, result.recognizedWords);
    }
  }

  Future<void> getUserDataForGemini() async {
    await FirebaseFirestore.instance
        .collection('users')
        .doc(UserAuth().userEmail)
        .get()
        .then((value) {
      final age = value.data()!['Age'];
      final alcoholConsumption = value.data()!['Alcohol Consumption'];
      final allergies = value.data()!['Allergies'];
      final avgHoursOfSleep = value.data()!['Avg. Hours of Sleep'];
      final diet = value.data()!['Diet'];
      final exerciseRoutine = value.data()!['Exercise Routine'];
      final familyHistory = value.data()!['Family History'];
      final gender = value.data()!['Gender'];
      final height = value.data()!['Height'];
      final medicalConditions = value.data()!['Medical Conditions'];
      final mentalHealthHistory = value.data()!['Mental Health History'];
      final physicalActivityHours = value.data()!['Physical Activity Hours'];
      final recreationalDrugUse = value.data()!['Recreational Drug Use'];
      final smokingHabits = value.data()!['Smoking Habits'];
      final stressManagement = value.data()!['Stress Management'];
      final weight = value.data()!['Weight'];
      final email = value.data()!['email'];
      final healthScore = value.data()!['healthScore'];
      final name = value.data()!['name'];
      userData.addAll({
        "age": age + " years",
        "alcoholConsumption": alcoholConsumption,
        "allergies": allergies,
        "avgHoursOfSleep": avgHoursOfSleep,
        "diet": diet,
        "exerciseRoutine": exerciseRoutine,
        "familyHistory": familyHistory,
        "gender": gender,
        "height": height + " cm",
        "medicalConditions": medicalConditions,
        "mentalHealthHistory": mentalHealthHistory,
        "physicalActivityHours": physicalActivityHours,
        "recreationalDrugUse": recreationalDrugUse,
        "smokingHabits": smokingHabits,
        "stressManagement": stressManagement,
        "weight": weight,
        "email": email,
        "healthScore": "$healthScore %",
        "name": name
      });
    });
    print(userData);
  }
}
