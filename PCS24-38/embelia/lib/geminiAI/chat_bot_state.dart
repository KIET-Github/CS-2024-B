part of 'chat_bot_cubit.dart';

@immutable
sealed class ChatBotState {}

final class ChatBotInitial extends ChatBotState {}

final class ChatBotListening extends ChatBotState {}

final class ChatBotLoading extends ChatBotState {}

final class ChatBotLoaded extends ChatBotState {
  final String message;

  ChatBotLoaded(this.message);
}

final class ChatBotError extends ChatBotState {
  final String message;

  ChatBotError(this.message);
}
