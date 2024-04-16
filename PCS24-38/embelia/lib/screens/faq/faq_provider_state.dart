part of 'faq_provider_cubit.dart';

@immutable
abstract class FaqProviderState {}

class FaqProviderInitial extends FaqProviderState {}

class FaqProviderLoading extends FaqProviderState {}

class FaqProviderLoaded extends FaqProviderState {
  final String answer;

  FaqProviderLoaded(this.answer);
}
