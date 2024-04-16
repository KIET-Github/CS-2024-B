import 'dart:convert';
import 'package:bloc/bloc.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../../constants.dart';
part 'faq_provider_state.dart';

class FaqProviderCubit extends Cubit<FaqProviderState> {
  FaqProviderCubit() : super(FaqProviderInitial());

  // Variable to store the answer
  static String _answer = "Mental Health";

  String get answer => _answer;

  // FUnction to get the answer from the API
  Future getFAQAnswer(question) async {
    emit(FaqProviderLoading());
    await http
        .get(
      Uri.parse("$uri/$question"),
    )
        .then(
      (value) {
        switch (value.statusCode) {
          case 200:
            emit(FaqProviderLoaded(jsonDecode(value.body)));
            _answer = jsonDecode(value.body);
            return _answer;
          case 404:
            _answer = "Not Found";
            return "Not Found";
          default:
            _answer = "Error";
            return "Error";
        }
      },
    ).catchError(
      (e) {
        _answer = "Error";
        return "Error";
      },
    );
  }
}
