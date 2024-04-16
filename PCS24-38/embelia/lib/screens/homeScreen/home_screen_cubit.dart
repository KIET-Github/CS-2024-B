import 'package:bloc/bloc.dart';
import 'package:meta/meta.dart';

part 'home_screen_state.dart';

class HomeScreenCubit extends Cubit<HomeScreenState> {
  HomeScreenCubit() : super(HomeScreenInitialTask());

  void changeTask() {
    if (state is HomeScreenInitialTask) {
      emit(HomeScreenRecommendedTask());
    } else {
      emit(HomeScreenInitialTask());
    }
  }
}
